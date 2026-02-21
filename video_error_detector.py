import cv2
import numpy as np
import csv
import os
import json
import matplotlib.pyplot as plt
import argparse
from skimage.metrics import structural_similarity as ssim


# ---------------------- LOAD VIDEO ----------------------

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, total_frames, width, height


# ---------------------- OPTICAL FLOW ----------------------

def compute_optical_flow(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)


# ---------------------- MAIN PROCESS ----------------------

def process_video(input_path, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap, fps, total_frames, width, height = load_video(input_path)

    out_path = os.path.join(output_dir, "output_processed.mp4")
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    csv_path = os.path.join(output_dir, "analysis_report.csv")
    json_path = os.path.join(output_dir, "summary.json")
    graph_path = os.path.join(output_dir, "motion_graph.png")

    prev_gray = None
    frame_idx = 0

    drop_count = 0
    merge_count = 0

    results = []
    flow_history = []
    diff_history = []

    print(f"Analyzing {input_path} at {fps} FPS...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Look ahead frame
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret_next, frame_next = cap.read()
        if ret_next:
            next_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
        else:
            next_gray = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        label = "Normal"
        color = (0, 255, 0)
        confidence = 0.0
        flow_mag = 0.0

        if prev_gray is not None and next_gray is not None:

            diff_prev = np.mean(cv2.absdiff(gray, prev_gray))
            diff_next = np.mean(cv2.absdiff(gray, next_gray))
            diff_prev_next = np.mean(cv2.absdiff(prev_gray, next_gray))

            diff_history.append(diff_prev)

            mean_diff = np.mean(diff_history)
            std_diff = np.std(diff_history)
            dynamic_threshold = mean_diff + 2 * std_diff

            # ---------------- DROP DETECTION ----------------
            if (
                diff_prev > dynamic_threshold and
                diff_next > dynamic_threshold and
                diff_prev_next < dynamic_threshold * 0.6
            ):
                label = "Frame Drop"
                color = (0, 0, 255)
                confidence = min(1.0, (diff_prev + diff_next) / 200)
                drop_count += 1

            else:
                # ---------------- MERGE DETECTION ----------------

                flow_mag = compute_optical_flow(prev_gray, gray)
                flow_history.append(flow_mag)

                if len(flow_history) > 10:
                    mean_flow = np.mean(flow_history)
                    std_flow = np.std(flow_history)
                else:
                    mean_flow = 0
                    std_flow = 0

                avg_adjacent = cv2.addWeighted(prev_gray, 0.5, next_gray, 0.5, 0)

                similarity_curr_avg, _ = ssim(gray, avg_adjacent, full=True)
                similarity_prev_next, _ = ssim(prev_gray, next_gray, full=True)

                # True merge characteristics:
                # 1. Current similar to blended neighbors
                # 2. Neighbors not similar to each other
                # 3. Motion within reasonable range

                if (
                    similarity_curr_avg > 0.88 and
                    similarity_prev_next < 0.85 and
                    mean_flow * 0.5 < flow_mag < mean_flow + 2 * std_flow
                ):
                    label = "Frame Merge"
                    color = (255, 0, 0)
                    confidence = similarity_curr_avg
                    merge_count += 1

        # -------- Visual Highlight --------
        if label != "Normal":
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, 10)

        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Timeline bar
        bar_width = int((frame_idx / total_frames) * width)
        cv2.rectangle(frame, (0, height - 20), (bar_width, height), color, -1)

        out.write(frame)

        results.append((frame_idx, label, round(confidence, 3)))

        prev_gray = gray
        frame_idx += 1

    cap.release()
    out.release()

    # CSV report
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Index", "Label", "Confidence"])
        writer.writerows(results)

    # JSON summary
    summary = {
        "total_frames": frame_idx,
        "frame_drops": drop_count,
        "frame_merges": merge_count,
        "corruption_percent": round(((drop_count + merge_count) / frame_idx) * 100, 2)
    }

    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=4)

    # Motion graph
    plt.figure(figsize=(10, 5))
    plt.plot(flow_history)
    plt.title("Optical Flow Magnitude")
    plt.xlabel("Frame")
    plt.ylabel("Flow")
    plt.grid(True)
    plt.savefig(graph_path)
    plt.close()

    print("\n========== FINAL REPORT ==========")
    print(f"Total Frames : {frame_idx}")
    print(f"Frame Drops  : {drop_count}")
    print(f"Frame Merges : {merge_count}")
    print("==================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Video Temporal Error Detector")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="output")

    args = parser.parse_args()
    process_video(args.input, args.output_dir)