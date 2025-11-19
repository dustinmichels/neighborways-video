from datetime import datetime
from typing import Optional

import cv2
from sqlmodel import Field, Session, SQLModel, create_engine

# Database configuration
DATABASE_URL = "sqlite:///detections.db"

# Define allowed labels
ALLOWED_LABELS = {"person", "bicycle", "car", "motorbike", "bus", "truck"}


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
    timestamp: datetime = Field(default_factory=datetime.utcnow)
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


def save_detections(
    results, frame, frame_number: int, min_confidence: float, crop_padding: int
):
    """Save all detections from a frame to the database"""
    with Session(engine) as session:
        # Get the detection results
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            saved_count = 0
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

                # Filter by allowed labels
                if label not in ALLOWED_LABELS:
                    continue

                # Filter by confidence
                if confidence < min_confidence:
                    continue

                # Extract bounding box coordinates (xyxy format)
                # Then, crop img with padding
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
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
                    label=label,
                    confidence=confidence,
                    bbox_x1=int(x1),
                    bbox_y1=int(y1),
                    bbox_x2=int(x2),
                    bbox_y2=int(y2),
                    frame_number=frame_number,
                    track_id=track_id,
                    img_data=img_bytes,
                )

                session.add(detection)
                saved_count += 1

            session.commit()
            print(f"Saved {saved_count} detections from frame {frame_number}")
