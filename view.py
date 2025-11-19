import cv2
import numpy as np
from sqlmodel import Session, select

from src.database import Detection, engine


def bytes_to_frame(img_bytes: bytes):
    """Convert bytes back to OpenCV frame"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


def view_first_image():
    """Retrieve and display the first image from the database"""
    with Session(engine) as session:
        # Query the first detection
        statement = select(Detection).order_by(Detection.id).limit(1)
        detection = session.exec(statement).first()

        if detection is None:
            print("No detections found in database")
            return

        # Convert bytes back to image
        img = bytes_to_frame(detection.img_data)

        if img is None:
            print("Failed to decode image")
            return

        # Display detection info
        print(f"Detection ID: {detection.id}")
        print(f"Label: {detection.label}")
        print(f"Confidence: {detection.confidence:.2f}")
        print(f"Frame: {detection.frame_number}")
        print(f"Track ID: {detection.track_id}")
        print(
            f"Bounding Box: ({detection.bbox_x1:.1f}, {detection.bbox_y1:.1f}) to ({detection.bbox_x2:.1f}, {detection.bbox_y2:.1f})"
        )
        print(f"Timestamp: {detection.timestamp}")

        # Display the image
        cv2.imshow(
            f"Detection: {detection.label} (confidence: {detection.confidence:.2f})",
            img,
        )
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    view_first_image()
