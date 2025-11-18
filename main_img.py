import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set

import cv2
import numpy as np
from rich.progress import (MofNCompleteColumn, Progress, SpinnerColumn,
                           TimeElapsedColumn)
from ultralytics import YOLO

from src.types import ImgRecord

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="YOLO object tracking and crop saving")
parser.add_argument(
    "--no-draw",
    action="store_true",
    help="Disable drawing boxes and displaying video feed (faster processing)",
)
args = parser.parse_args()

# --- CONFIG ---
# MODEL_PATH = "yolov8s.pt"
MODEL_PATH = "yolov10n.pt"
# VIDEO_PATH = "video/glen-oliver/after_glen-oliver.mp4"
VIDEO_PATH = "video/glen-oliver/short/after_glen-oliver.mp4"
OUTPUT_DIR = Path("out/saved_unique_crops")
IMG_DIR = OUTPUT_DIR / "img"
MIN_CONFIDENCE = 0.5  # only consider detections above this confidence
CROP_PADDING = 10  # pixels of padding around the bbox
CROP_SIZE = (256, 256)  # saved crop size (width, height), or None to keep original
ENABLE_DRAWING = not args.no_draw  # Control drawing based on command line arg
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
# ----------------

# Delete output dir if it exists
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# List to store ImgRecord instances
manifest_records: List[ImgRecord] = []

# Load YOLO model with verbose=False to suppress output
model = YOLO(MODEL_PATH, verbose=False)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

# Get total frame count
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

# Create progress bar (only when not drawing to screen)
if not ENABLE_DRAWING:
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    progress.start()
    task = progress.add_task("[cyan]Processing frames...", total=total_frames)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    # Skip frames for faster processing
    if frame_no % PROCESS_EVERY_N_FRAMES != 0:
        if not ENABLE_DRAWING:
            progress.update(task, advance=1)  # Still update progress bar
        continue

    # Run tracking with verbose=False to suppress output
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

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

                # Create ImgRecord and add to manifest
                record = ImgRecord(
                    saved_path=str(save_path),
                    label=label,
                    track_id=track_id,
                    frame_no=frame_no,
                    conf=conf,
                )
                manifest_records.append(record)

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

    # Update progress bar or display video
    if ENABLE_DRAWING:
        cv2.imshow("Tracking and Saving Unique Crops", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Update progress bar
        progress.update(task, advance=1)

cap.release()
if ENABLE_DRAWING:
    cv2.destroyAllWindows()
else:
    progress.stop()

print(f"\nSaved crops to: {OUTPUT_DIR.resolve()}")

# Write CSV manifest from ImgRecord objects
print("Writing CSV manifest...")
manifest_csv_path = OUTPUT_DIR / "manifest.csv"
with open(manifest_csv_path, "w", newline="") as csvfile:
    if manifest_records:
        # Use the field names from the Pydantic model
        fieldnames = list(ImgRecord.model_fields.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in manifest_records:
            writer.writerow(record.model_dump())

print(f"CSV manifest saved to: {manifest_csv_path.resolve()}")

# Write JSON manifest from ImgRecord objects
print("Writing JSON manifest...")
json_manifest_path = OUTPUT_DIR / "manifest.json"
with open(json_manifest_path, "w") as jsonfile:
    # Convert all records to dictionaries
    manifest_data = [record.model_dump() for record in manifest_records]
    json.dump(manifest_data, jsonfile, indent=2)

print(f"JSON manifest saved to: {json_manifest_path.resolve()}")
print(f"Total unique objects saved: {len(manifest_records)}")
