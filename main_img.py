import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Set

import cv2
import numpy as np
from ultralytics import YOLO

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="YOLO object tracking and crop saving")
parser.add_argument(
    "--no-draw",
    action="store_true",
    help="Disable drawing boxes and displaying video feed (faster processing)",
)
args = parser.parse_args()

# --- CONFIG ---
MODEL_PATH = "yolov8s.pt"
VIDEO_PATH = "video/glen-oliver/short/before_glen-oliver.mp4"
OUTPUT_DIR = Path("out/saved_unique_crops")
IMG_DIR = OUTPUT_DIR / "img"
MIN_CONFIDENCE = 0.3  # only consider detections above this confidence
CROP_PADDING = 10  # pixels of padding around the bbox
CROP_SIZE = (256, 256)  # saved crop size (width, height), or None to keep original
ENABLE_DRAWING = not args.no_draw  # Control drawing based on command line arg
# ----------------

# Delete output dir if it exists
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# CSV manifest
manifest_path = OUTPUT_DIR / "manifest.csv"
manifest_file = open(manifest_path, "w", newline="")
csv_writer = csv.writer(manifest_file)
csv_writer.writerow(
    [
        "saved_path",
        "label",
        "track_id",
        "frame_no",
        "conf",
    ]
)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

# Track seen IDs per class
unique_objects: Dict[str, Set[int]] = {
    "person": set(),
    "bicycle": set(),
    "car": set(),
    "motorbike": set(),
    "bus": set(),
    "truck": set(),
}

# Colors for on-screen drawing
colors = {
    "person": (0, 255, 0),  # Bright Green
    "bicycle": (255, 0, 255),  # Magenta
    "car": (255, 0, 0),  # Red
    "motorbike": (255, 165, 0),  # Orange
    "bus": (0, 255, 255),  # Cyan
    "truck": (128, 0, 255),  # Purple
}

frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    # Run tracking (persist=True so tracker keeps IDs across frames)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue

        boxes = r.boxes
        for i in range(len(boxes)):
            box = boxes[i]
            try:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = (
                    int(box.id[0]) if getattr(box, "id", None) is not None else None
                )
                xyxy = (
                    box.xyxy[0].cpu().numpy()
                    if hasattr(box.xyxy[0], "cpu")
                    else np.array(box.xyxy[0])
                )
            except Exception:
                continue

            if conf < MIN_CONFIDENCE:
                continue

            label = model.names.get(cls_id, str(cls_id))
            if label not in unique_objects:
                continue

            if track_id is None:
                continue

            # Save only the first appearance of each unique track ID per class
            if track_id not in unique_objects[label]:
                unique_objects[label].add(track_id)

                x1, y1, x2, y2 = map(int, xyxy)
                h, w = frame.shape[:2]

                # Ensure the entire bounding box (plus padding) is captured safely
                x1p = max(0, min(x1 - CROP_PADDING, w - 1))
                y1p = max(0, min(y1 - CROP_PADDING, h - 1))
                x2p = max(0, min(x2 + CROP_PADDING, w))
                y2p = max(0, min(y2 + CROP_PADDING, h))

                # Defensive check
                if x2p <= x1p or y2p <= y1p:
                    continue

                crop = frame[y1p:y2p, x1p:x2p].copy()
                if crop.size == 0:
                    continue

                # Optionally resize the crop
                if CROP_SIZE is not None:
                    crop = cv2.resize(crop, CROP_SIZE)

                # Generate filename with frame number first, then label
                filename = f"f{frame_no:06d}_{label}_id{track_id}_c{conf:.2f}.jpg"
                save_path = IMG_DIR / filename

                # Save the cropped image
                cv2.imwrite(str(save_path), crop)

                # Record in CSV manifest
                csv_writer.writerow(
                    [
                        str(save_path),
                        label,
                        track_id,
                        frame_no,
                        f"{conf:.4f}",
                    ]
                )
                manifest_file.flush()

                # Optional visual cue on the frame (only if drawing enabled)
                if ENABLE_DRAWING:
                    cv2.putText(
                        frame,
                        f"SAVED {label} ID:{track_id}",
                        (x1, max(15, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # Draw bbox on display frame (only if drawing enabled)
            if ENABLE_DRAWING:
                color = colors.get(label, (0, 255, 0))
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ID:{track_id} {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

    # Show per-class counts and display video (only if drawing enabled)
    if ENABLE_DRAWING:
        y_offset = 30
        for label, ids in unique_objects.items():
            color = colors.get(label, (0, 255, 255))
            cv2.putText(
                frame,
                f"{label}: {len(ids)}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            y_offset += 28

        # Display live video feed
        cv2.imshow("Tracking and Saving Unique Crops", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Print progress periodically when not displaying
        if frame_no % 100 == 0:
            print(f"Processed {frame_no} frames...")

cap.release()
if ENABLE_DRAWING:
    cv2.destroyAllWindows()
manifest_file.close()

# Print summary
print("Final unique object counts:")
for k, v in unique_objects.items():
    print(f"{k}: {len(v)}")
print(f"Saved crops + manifest to: {OUTPUT_DIR.resolve()}")

# Convert CSV manifest to JSON
print("Converting manifest to JSON...")
json_manifest_path = OUTPUT_DIR / "manifest.json"

manifest_data = []
with open(manifest_path, "r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        row["track_id"] = int(row["track_id"])
        row["frame_no"] = int(row["frame_no"])
        row["conf"] = float(row["conf"])
        manifest_data.append(row)

with open(json_manifest_path, "w") as jsonfile:
    json.dump(manifest_data, jsonfile, indent=2)

print(f"JSON manifest saved to: {json_manifest_path.resolve()}")
