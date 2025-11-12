import csv
import shutil
import time
from pathlib import Path
from typing import Dict, Set

import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = "yolov8s.pt"
VIDEO_PATH = "video/glen-oliver/short/before_glen-oliver.mp4"
OUTPUT_DIR = Path("out/saved_unique_crops")
MIN_CONFIDENCE = 0.3  # only consider detections above this confidence
CROP_PADDING = 10  # pixels of padding around the bbox
CROP_SIZE = (256, 256)  # saved crop size (width, height), or None to keep original
# ----------------

# Delete output dir if it exists
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "timestamp",
    ]
)

# Load model
model = YOLO(MODEL_PATH)

# Video
cap = cv2.VideoCapture(VIDEO_PATH)

# Track seen IDs per class (so we only save one crop per unique track id per class)
unique_objects: Dict[str, Set[int]] = {
    "person": set(),
    "bicycle": set(),
    "car": set(),
    "motorbike": set(),
    "bus": set(),
    "truck": set(),
}

# Colors for display (optional)
colors = {
    "person": (0, 255, 0),  # Bright Green
    "bicycle": (255, 0, 255),  # Magenta
    "car": (255, 0, 0),  # Red
    "motorbike": (255, 165, 0),  # Orange (changed from blue-ish)
    "bus": (0, 255, 255),  # Cyan
    "truck": (128, 0, 255),  # Purple (changed from similar to motorbike)
}

frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    # Run tracking (persist True so tracker keeps IDs)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue

        boxes = r.boxes
        for i in range(len(boxes)):
            box = boxes[i]

            # convert to python types
            # Some attributes are tensors - cast safely
            try:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                # id attr may not exist in old API; handle gracefully
                track_id = (
                    int(box.id[0]) if getattr(box, "id", None) is not None else None
                )
                xyxy = (
                    box.xyxy[0].cpu().numpy()
                    if hasattr(box.xyxy[0], "cpu")
                    else np.array(box.xyxy[0])
                )
            except Exception:
                # best-effort fallback if API differs
                continue

            if conf < MIN_CONFIDENCE:
                continue

            label = model.names.get(cls_id, str(cls_id))

            # Only consider classes we're tracking (matches your original dict)
            if label not in unique_objects:
                continue

            if track_id is None:
                # If no track id (shouldn't happen with a tracking call), skip
                continue

            # If this is the first time we see this track id for this label -> save crop + metadata
            if track_id not in unique_objects[label]:
                unique_objects[label].add(track_id)

                x1, y1, x2, y2 = map(int, xyxy)
                h, w = frame.shape[:2]

                # Apply padding and clamp to image bounds
                x1p = max(0, x1 - CROP_PADDING)
                y1p = max(0, y1 - CROP_PADDING)
                x2p = min(w - 1, x2 + CROP_PADDING)
                y2p = min(h - 1, y2 + CROP_PADDING)

                crop = frame[y1p:y2p, x1p:x2p].copy()
                if crop.size == 0:
                    # safety: skip invalid crop
                    continue

                # Optionally resize
                if CROP_SIZE is not None:
                    crop = cv2.resize(crop, CROP_SIZE)

                # Create class folder
                class_dir = OUTPUT_DIR / label
                class_dir.mkdir(parents=True, exist_ok=True)

                # Build safe filename
                timestamp = int(time.time())
                filename = (
                    f"{label}_id{track_id}_f{frame_no}_c{conf:.2f}_{timestamp}.jpg"
                )
                save_path = class_dir / filename

                # Save crop
                cv2.imwrite(str(save_path), crop)

                # Write manifest row
                csv_writer.writerow(
                    [
                        str(save_path),
                        label,
                        track_id,
                        frame_no,
                        f"{conf:.4f}",
                        x1p,
                        y1p,
                        x2p,
                        y2p,
                        timestamp,
                    ]
                )
                manifest_file.flush()  # flush incrementally so partial results are saved if interrupted

                # OPTIONAL: draw a small thumbnail or marker on the frame to show it was saved
                cv2.putText(
                    frame,
                    f"SAVED {label} ID:{track_id}",
                    (x1, max(15, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # draw bbox on frame (for display)
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

    # Display counts on-screen
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

    cv2.imshow("Tracking and Saving Unique Crops", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
manifest_file.close()

print("Final unique object counts:")
for k, v in unique_objects.items():
    print(f"{k}: {len(v)}")
print(f"Saved crops + manifest to: {OUTPUT_DIR.resolve()}")
