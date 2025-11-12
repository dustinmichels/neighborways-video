from typing import Dict, Set

import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Start video
cap = cv2.VideoCapture("video/glen-oliver/short/before_glen-oliver.mp4")

# Dictionary to store unique IDs per class
unique_objects: Dict[str, Set[int]] = {
    "person": set(),
    "car": set(),
    "motorbike": set(),
    "bicycle": set(),
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

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

    # Display current counts
    y_offset = 30
    for label, ids in unique_objects.items():
        cv2.putText(
            frame,
            f"{label}: {len(ids)}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
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
