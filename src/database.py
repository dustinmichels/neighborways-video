from datetime import datetime
from typing import Optional

import cv2
from sqlmodel import Field, Session, SQLModel, create_engine

# Database configuration
DATABASE_URL = "sqlite:///detections.db"


# Define the Detection model
class Detection(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    label: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    frame_number: int
    track_id: Optional[int] = None
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
            for i, box in enumerate(boxes):
                # Extract confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                label = results[0].names[class_id]

                # Filter by confidence
                if confidence < min_confidence:
                    continue

                # Extract bounding box coordinates (xyxy format)
                # Then, crop img with padding
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1_crop = max(int(x1) - crop_padding, 0)
                y1_crop = max(int(y1) - crop_padding, 0)
                x2_crop = min(int(x2) + crop_padding, frame.shape[1])
                y2_crop = min(int(y2) + crop_padding, frame.shape[0])
                cropped_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                img_bytes = frame_to_bytes(cropped_img)

                # Extract track ID if available
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())

                # Create detection record
                detection = Detection(
                    label=label,
                    confidence=confidence,
                    bbox_x1=float(x1),
                    bbox_y1=float(y1),
                    bbox_x2=float(x2),
                    bbox_y2=float(y2),
                    frame_number=frame_number,
                    track_id=track_id,
                    img_data=img_bytes,
                )

                session.add(detection)

            session.commit()
            print(f"Saved {len(boxes)} detections from frame {frame_number}")
