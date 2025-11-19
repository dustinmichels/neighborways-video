import os
from pathlib import Path

import cv2
import numpy as np
from sqlmodel import Session, select

from src.database import Detection, engine


def bytes_to_frame(img_bytes: bytes):
    """Convert bytes back to OpenCV frame"""
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


def sanitize_filename(filename: str) -> str:
    """Remove or replace characters that might cause issues in filenames"""
    # Replace spaces and other problematic characters
    return filename.replace(" ", "_").replace("/", "_").replace("\\", "_")


def export_images():
    """Export all images from the database to individual files"""
    # Create output directory
    output_dir = Path("out/fromdb")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting images to: {output_dir.absolute()}")

    with Session(engine) as session:
        # Query all detections
        statement = select(Detection).order_by(Detection.frame_number, Detection.id)
        detections = session.exec(statement).all()

        if not detections:
            print("No detections found in database")
            return

        print(f"Found {len(detections)} detections to export")

        exported_count = 0
        failed_count = 0

        for detection in detections:
            # Convert bytes back to image
            img = bytes_to_frame(detection.img_data)

            if img is None:
                print(f"Failed to decode image for detection ID {detection.id}")
                failed_count += 1
                continue

            # Create filename: {frame}_{id}_{label}.jpg
            label = sanitize_filename(detection.label)
            filename = f"{detection.frame_number}_{detection.id}_{label}.jpg"
            filepath = output_dir / filename

            # Save the image
            success = cv2.imwrite(str(filepath), img)

            if success:
                exported_count += 1
                if exported_count % 10 == 0:  # Progress update every 10 images
                    print(f"Exported {exported_count}/{len(detections)} images...")
            else:
                print(f"Failed to save image: {filename}")
                failed_count += 1

        print(f"\nExport complete!")
        print(f"Successfully exported: {exported_count} images")
        if failed_count > 0:
            print(f"Failed: {failed_count} images")
        print(f"Output location: {output_dir.absolute()}")


if __name__ == "__main__":
    export_images()
