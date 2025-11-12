import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy

LABELS = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "person",
    "truck",
]


# Open video file or webcam
cap = cv2.VideoCapture("video/glen-oliver/short/before_glen-oliver.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Extract detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # Only detect cars, bikes, and persons
            if label in LABELS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

    # Display
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
