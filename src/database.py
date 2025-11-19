from datetime import datetime, timezone
from typing import Optional

import cv2
from sqlmodel import Field, Session, SQLModel, create_engine

# Database configuration
DATABASE_URL = "sqlite:///detections.db"

# Define allowed labels
ALLOWED_LABELS = {"person", "bicycle", "car", "motorbike", "bus", "truck", "cyclist"}


# Define the Detection model
class Detection(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: int
    label: str
    confidence: float
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    frame_number: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    img_data: bytes

    class Config:
        arbitrary_types_allowed = True


# Create database engine and tables
engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    """Initialize database tables"""
    SQLModel.metadata.create_all(engine)


def frame_to_bytes(frame) -> bytes:
    """Convert OpenCV frame to bytes"""
    success, buffer = cv2.imencode(".jpg", frame)
    if success:
        return buffer.tobytes()
    return b""


def boxes_touching(box1, box2, threshold=10):
    """Check if two bounding boxes are touching or overlapping

    Args:
        box1: tuple of (x1, y1, x2, y2)
        box2: tuple of (x1, y1, x2, y2)
        threshold: pixel distance to consider boxes as touching

    Returns:
        bool: True if boxes are touching or overlapping
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Check if boxes overlap or are within threshold distance
    horizontal_gap = max(0, max(x1_1, x1_2) - min(x2_1, x2_2))
    vertical_gap = max(0, max(y1_1, y1_2) - min(y2_1, y2_2))

    return horizontal_gap <= threshold and vertical_gap <= threshold


def create_union_box(box1, box2):
    """Create a bounding box that encompasses both input boxes

    Args:
        box1: tuple of (x1, y1, x2, y2)
        box2: tuple of (x1, y1, x2, y2)

    Returns:
        tuple: (x1, y1, x2, y2) of the union box
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    return (min(x1_1, x1_2), min(y1_1, y1_2), max(x2_1, x2_2), max(y2_1, y2_2))


def save_detections(
    results, frame, frame_number: int, min_confidence: float, crop_padding: int
):
    """Save all detections from a frame to the database"""
    with Session(engine) as session:
        # Get the detection results
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            saved_count = 0

            # Store all detections first
            detections_list = []

            for _, box in enumerate(boxes):
                # Extract basic info
                try:
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    label = results[0].names[class_id]
                    track_id = int(box.id[0].cpu().numpy())
                except Exception as e:
                    print(f"Error extracting detection data: {e}")
                    continue

                # Filter by confidence
                if confidence < min_confidence:
                    continue

                # Extract bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                detections_list.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "track_id": track_id,
                    }
                )

            # Find person-bicycle pairs and create cyclist detections
            cyclist_pairs = set()  # Track which detections are used in cyclist pairs

            for i, det1 in enumerate(detections_list):
                if det1["label"] == "person":
                    for j, det2 in enumerate(detections_list):
                        if det2["label"] == "bicycle" and i != j:
                            if boxes_touching(det1["bbox"], det2["bbox"]):
                                # Create cyclist detection
                                union_box = create_union_box(det1["bbox"], det2["bbox"])
                                x1, y1, x2, y2 = union_box

                                # Crop image with padding
                                x1_crop = max(x1 - crop_padding, 0)
                                y1_crop = max(y1 - crop_padding, 0)
                                x2_crop = min(x2 + crop_padding, frame.shape[1])
                                y2_crop = min(y2 + crop_padding, frame.shape[0])
                                cropped_img = frame[
                                    int(y1_crop) : int(y2_crop),
                                    int(x1_crop) : int(x2_crop),
                                ]
                                img_bytes = frame_to_bytes(cropped_img)

                                # Use average confidence and person's track_id
                                avg_confidence = (
                                    det1["confidence"] + det2["confidence"]
                                ) / 2

                                # Create cyclist detection
                                detection = Detection(
                                    label="cyclist",
                                    confidence=avg_confidence,
                                    bbox_x1=x1,
                                    bbox_y1=y1,
                                    bbox_x2=x2,
                                    bbox_y2=y2,
                                    frame_number=frame_number,
                                    track_id=det1["track_id"],  # Use person's track_id
                                    img_data=img_bytes,
                                )

                                session.add(detection)
                                saved_count += 1

                                # Mark these detections as used in a cyclist pair
                                cyclist_pairs.add(i)
                                cyclist_pairs.add(j)

            # Save individual detections
            for i, det in enumerate(detections_list):
                # Skip individual person/bicycle if they were combined into cyclist
                # Comment out these 2 lines if you want to save both individual AND cyclist detections
                if i in cyclist_pairs:
                    continue

                # Filter by allowed labels
                if det["label"] not in ALLOWED_LABELS:
                    continue

                x1, y1, x2, y2 = det["bbox"]

                # Crop image with padding
                x1_crop = max(x1 - crop_padding, 0)
                y1_crop = max(y1 - crop_padding, 0)
                x2_crop = min(x2 + crop_padding, frame.shape[1])
                y2_crop = min(y2 + crop_padding, frame.shape[0])
                cropped_img = frame[
                    int(y1_crop) : int(y2_crop), int(x1_crop) : int(x2_crop)
                ]
                img_bytes = frame_to_bytes(cropped_img)

                # Create detection record
                detection = Detection(
                    label=det["label"],
                    confidence=det["confidence"],
                    bbox_x1=x1,
                    bbox_y1=y1,
                    bbox_x2=x2,
                    bbox_y2=y2,
                    frame_number=frame_number,
                    track_id=det["track_id"],
                    img_data=img_bytes,
                )

                session.add(detection)
                saved_count += 1

            session.commit()
            # print(f"Saved {saved_count} detections from frame {frame_number}")
