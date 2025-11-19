import os
from collections import defaultdict
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


def calculate_sharpness(img):
    """Calculate image sharpness using Laplacian variance"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def sanitize_filename(filename: str) -> str:
    """Remove or replace characters that might cause issues in filenames"""
    return filename.replace(" ", "_").replace("/", "_").replace("\\", "_")


def select_best_images():
    """For each track_id, select and save the highest quality image"""
    # Create output directory
    output_dir = Path("out/best")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selecting best images and saving to: {output_dir.absolute()}")

    with Session(engine) as session:
        # Query all detections
        statement = select(Detection).order_by(
            Detection.track_id, Detection.frame_number
        )
        detections = session.exec(statement).all()

        if not detections:
            print("No detections found in database")
            return

        print(f"Found {len(detections)} total detections")

        # Group detections by track_id
        tracks = defaultdict(list)
        for detection in detections:
            tracks[detection.track_id].append(detection)

        print(f"Found {len(tracks)} unique tracks")

        saved_count = 0

        # Process each track
        for track_id, track_detections in tracks.items():
            print(
                f"\nProcessing track_id {track_id} ({len(track_detections)} images)..."
            )

            best_detection = None
            best_sharpness = -1
            best_img = None

            # Evaluate each image in this track
            for detection in track_detections:
                img = bytes_to_frame(detection.img_data)

                if img is None:
                    print(f"  Failed to decode image for detection ID {detection.id}")
                    continue

                # Calculate sharpness
                sharpness = calculate_sharpness(img)
                print(f"  Frame {detection.frame_number}: sharpness = {sharpness:.2f}")

                # Keep track of the best one
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_detection = detection
                    best_img = img

            # Save the best image for this track
            if best_img is not None and best_detection is not None:
                label = sanitize_filename(best_detection.label)
                filename = f"track_{track_id}_{label}.jpg"
                filepath = output_dir / filename

                success = cv2.imwrite(str(filepath), best_img)

                if success:
                    saved_count += 1
                    print(f"  ✓ Saved best image: {filename}")
                    print(
                        f"    (Frame {best_detection.frame_number}, sharpness: {best_sharpness:.2f})"
                    )
                else:
                    print(f"  ✗ Failed to save: {filename}")
            else:
                print(f"  ✗ No valid images found for track_id {track_id}")

        print(f"\n{'=' * 60}")
        print(f"Processing complete!")
        print(f"Saved {saved_count}/{len(tracks)} best images")
        print(f"Output location: {output_dir.absolute()}")


if __name__ == "__main__":
    select_best_images()
