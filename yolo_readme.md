# YOLO Detection Database

Save YOLO object detections to SQLite database with full frame images.

## Features

- **Pydantic/SQLModel** for data validation and ORM
- Saves each detection with:
  - Auto-incrementing ID
  - Object label (class name)
  - Confidence score
  - Bounding box coordinates (x1, y1, x2, y2)
  - Frame number
  - Timestamp
  - Full frame image (as bytes)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run Detection and Save to Database

```bash
python yolo_detection_db.py
```

This will:

- Process your video file
- Detect objects using YOLO11
- Save each detection to `detections.db`
- Display live tracking visualization

### 2. Query and View Detections

```bash
python query_detections.py
```

This provides helper functions to:

- View statistics about saved detections
- Query detections by label, confidence, or frame
- Retrieve and display specific detections with bounding boxes

## Database Schema

```python
class Detection(SQLModel, table=True):
    id: Optional[int]           # Primary key
    label: str                  # Object class name
    confidence: float           # Detection confidence (0-1)
    bbox_x1: float             # Bounding box top-left x
    bbox_y1: float             # Bounding box top-left y
    bbox_x2: float             # Bounding box bottom-right x
    bbox_y2: float             # Bounding box bottom-right y
    frame_number: int          # Frame index in video
    timestamp: datetime        # When detection was saved
    img_data: bytes            # Full frame as JPEG bytes
```

## Example Queries

```python
from sqlmodel import Session, select, create_engine
from yolo_detection_db import Detection

engine = create_engine("sqlite:///detections.db")

with Session(engine) as session:
    # Get all person detections with high confidence
    statement = select(Detection).where(
        Detection.label == "person",
        Detection.confidence >= 0.8
    )
    results = session.exec(statement).all()

    # Get detections from specific frame
    frame_detections = session.exec(
        select(Detection).where(Detection.frame_number == 100)
    ).all()
```

## Storage Considerations

**Note:** Storing full images for every detection can create large databases. Consider:

1. **Store cropped objects** instead of full frames:

```python
# Crop the detected object
x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
cropped = frame[y1:y2, x1:x2]
img_bytes = frame_to_bytes(cropped)
```

2. **Sample frames** (save every Nth frame):

```python
if frame_number % 10 == 0:  # Save every 10th frame
    save_detections(results, frame, frame_number)
```

3. **Store only high-confidence detections**:

```python
if confidence >= 0.7:
    # Save detection
```

## Output

- Database file: `detections.db` (SQLite)
- Query results in console
- Visual display of detections with bounding boxes
