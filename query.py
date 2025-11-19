import cv2
import numpy as np
from sqlmodel import Session, create_engine, select

from yolo_db import Detection

# Create database engine
DATABASE_URL = "sqlite:///detections.db"
engine = create_engine(DATABASE_URL, echo=False)


def bytes_to_frame(img_bytes: bytes):
    """Convert bytes back to OpenCV frame"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


def view_detection(detection_id: int):
    """View a specific detection by ID"""
    with Session(engine) as session:
        detection = session.get(Detection, detection_id)

        if detection:
            print(f"ID: {detection.id}")
            print(f"Label: {detection.label}")
            print(f"Confidence: {detection.confidence:.2f}")
            print(
                f"Bounding Box: ({detection.bbox_x1:.1f}, {detection.bbox_y1:.1f}, "
                f"{detection.bbox_x2:.1f}, {detection.bbox_y2:.1f})"
            )
            print(f"Frame: {detection.frame_number}")
            print(f"Timestamp: {detection.timestamp}")

            # Display the image
            frame = bytes_to_frame(detection.img_data)
            if frame is not None:
                # Draw bounding box on the frame
                cv2.rectangle(
                    frame,
                    (int(detection.bbox_x1), int(detection.bbox_y1)),
                    (int(detection.bbox_x2), int(detection.bbox_y2)),
                    (0, 255, 0),
                    2,
                )

                # Add label
                label_text = f"{detection.label} {detection.confidence:.2f}"
                cv2.putText(
                    frame,
                    label_text,
                    (int(detection.bbox_x1), int(detection.bbox_y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(f"Detection {detection.id}", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"Detection {detection_id} not found")


def query_detections(
    label: str = None,
    min_confidence: float = 0.0,
    frame_number: int = None,
    limit: int = 10,
):
    """Query detections with filters"""
    with Session(engine) as session:
        statement = select(Detection)

        if label:
            statement = statement.where(Detection.label == label)

        if min_confidence > 0:
            statement = statement.where(Detection.confidence >= min_confidence)

        if frame_number is not None:
            statement = statement.where(Detection.frame_number == frame_number)

        statement = statement.limit(limit)

        results = session.exec(statement).all()

        print(f"Found {len(results)} detections:")
        for det in results:
            print(
                f"ID: {det.id}, Label: {det.label}, Confidence: {det.confidence:.2f}, "
                f"Frame: {det.frame_number}"
            )

        return results


def get_statistics():
    """Get statistics about the detections"""
    with Session(engine) as session:
        # Total detections
        total = session.exec(select(Detection)).all()
        print(f"Total detections: {len(total)}")

        # Detections per label
        labels = {}
        for det in total:
            labels[det.label] = labels.get(det.label, 0) + 1

        print("\nDetections per label:")
        for label, count in sorted(labels.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}")

        # Frame count
        frames = set(det.frame_number for det in total)
        print(f"\nTotal frames with detections: {len(frames)}")


if __name__ == "__main__":
    # Example usage
    print("=== Database Statistics ===")
    get_statistics()

    print("\n=== Query Examples ===")

    # Query all detections with confidence > 0.8
    print("\nHigh confidence detections (>0.8):")
    query_detections(min_confidence=0.8, limit=5)

    # Query specific label
    print("\nPerson detections:")
    query_detections(label="person", limit=5)

    # View first detection
    print("\nViewing first detection...")
    view_detection(1)
