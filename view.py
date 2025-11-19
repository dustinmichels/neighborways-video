from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from sqlmodel import Field, Session, SQLModel, create_engine, select


# Detection model (same as your original)
class Detection(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    label: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    frame_number: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    img_data: bytes


engine = create_engine("sqlite:///detections.db", echo=False)

# Get and display first image
with Session(engine) as session:
    detection = session.exec(select(Detection).limit(1)).first()

    if detection:
        # Convert bytes to image
        nparr = np.frombuffer(detection.img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Draw bounding box
        cv2.rectangle(
            img,
            (int(detection.bbox_x1), int(detection.bbox_y1)),
            (int(detection.bbox_x2), int(detection.bbox_y2)),
            (0, 255, 0),
            2,
        )

        # Show image
        cv2.imshow(f"{detection.label}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
