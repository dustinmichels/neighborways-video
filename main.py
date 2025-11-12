from typing import Dict, Set

import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Start video
# cap = cv2.VideoCapture("video/glen-oliver/short-resample/before_glen-oliver.mp4")
cap = cv2.VideoCapture("video/glen-oliver/short/before_glen-oliver.mp4")

# Dictionary to store unique IDs per class
unique_objects: Dict[str, Set[int]] = {
    "person": set(),
    "bicycle": set(),
    "car": set(),
    "motorbike": set(),
    "bus": set(),
    "truck": set(),
}

# Define colors for each object class (BGR format)
colors = {
    "person": (0, 255, 0),  # Green
    "bicycle": (255, 0, 255),  # Magenta
    "car": (255, 0, 0),  # Blue
    "motorbike": (0, 165, 255),  # Orange
    "bus": (0, 255, 255),  # Yellow
    "truck": (0, 140, 255),  # Dark Orange
    "train": (255, 255, 0),  # Cyan
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking (not just detection)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    for r in results:
        if not hasattr(r, "boxes"):
            continue

        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # Each object now has a unique tracking ID
            track_id = int(box.id[0]) if box.id is not None else None

            if label in unique_objects.keys() and track_id is not None:
                # Add track ID to the set for that class
                unique_objects[label].add(track_id)

                # Get color for this object class
                color = colors.get(label, (0, 255, 0))  # Default to green if not found

                # Draw bounding box with class-specific color
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,  # Use same color for text
                    2,
                )

    # Display current counts with class-specific colors
    y_offset = 30
    for label, ids in unique_objects.items():
        color = colors.get(label, (0, 255, 255))
        cv2.putText(
            frame,
            f"{label}: {len(ids)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,  # Use class color instead of cyan
            2,
        )
        y_offset += 30

    cv2.imshow("Tracking and Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Final unique object counts:")
for k, v in unique_objects.items():
    print(f"{k}: {len(v)}")
