import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sqlmodel import Field, Session, SQLModel, create_engine, select

from src.database import Detection, engine


class CombinedDetection(SQLModel, table=True):
    """Table to store the best image and metadata for each track"""

    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: int = Field(index=True, unique=True)
    start_frame: int
    end_frame: int
    avg_conf: float
    unique_labels: str  # Stored as comma-separated string
    label: str  # Majority label
    img_data: bytes  # Best image data


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


def normalize_scores(scores):
    """Normalize a list of scores to 0-1 range"""
    if not scores or len(scores) == 1:
        return [1.0] * len(scores)

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def calculate_composite_score(
    sharpness: float,
    confidence: float,
    dimensions: int,
    sharpness_weight: float = 0.4,
    confidence_weight: float = 0.3,
    dimensions_weight: float = 0.3,
):
    """
    Calculate composite score from normalized metrics.
    Default weights: sharpness=0.4, confidence=0.3, dimensions=0.3
    """
    return (
        sharpness * sharpness_weight
        + confidence * confidence_weight
        + dimensions * dimensions_weight
    )


def sanitize_filename(filename: str) -> str:
    """Remove or replace characters that might cause issues in filenames"""
    return filename.replace(" ", "_").replace("/", "_").replace("\\", "_")


def select_best_images(
    sharpness_weight: float = 0.4,
    confidence_weight: float = 0.3,
    dimensions_weight: float = 0.3,
):
    """
    For each track_id, select the highest quality image and save track data to database.

    Args:
        sharpness_weight: Weight for sharpness metric (default: 0.4)
        confidence_weight: Weight for confidence metric (default: 0.3)
        dimensions_weight: Weight for dimensions metric (default: 0.3)
    """
    # Create the Track table if it doesn't exist
    SQLModel.metadata.create_all(engine)

    # Create output directory for image files
    output_dir = Path("out/best")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selecting best images and saving to database and {output_dir.absolute()}")
    print(
        f"Weights: sharpness={sharpness_weight}, confidence={confidence_weight}, dimensions={dimensions_weight}"
    )

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

            # Collect metrics for all images in this track
            metrics = []

            # Aggregate statistics for the track
            confidences = []
            labels = []
            frame_numbers = []

            for detection in track_detections:
                img = bytes_to_frame(detection.img_data)

                if img is None:
                    print(f"  Failed to decode image for detection ID {detection.id}")
                    continue

                # Calculate metrics
                sharpness = calculate_sharpness(img)
                confidence = detection.confidence
                height, width = img.shape[:2]
                dimensions = width * height

                metrics.append(
                    {
                        "detection": detection,
                        "img": img,
                        "sharpness": sharpness,
                        "confidence": confidence,
                        "dimensions": dimensions,
                    }
                )

                # Collect stats for aggregation
                confidences.append(confidence)
                labels.append(detection.label)
                frame_numbers.append(detection.frame_number)

            if not metrics:
                print(f"  ✗ No valid images found for track_id {track_id}")
                continue

            # Calculate aggregated statistics
            start_frame = min(frame_numbers)
            end_frame = max(frame_numbers)
            avg_conf = sum(confidences) / len(confidences)
            unique_labels_list = list(set(labels))
            unique_labels_str = ",".join(sorted(unique_labels_list))

            # Get majority label
            label_counts = Counter(labels)
            majority_label = label_counts.most_common(1)[0][0]

            # Normalize each metric across all images in this track
            sharpness_scores = [m["sharpness"] for m in metrics]
            confidence_scores = [m["confidence"] for m in metrics]
            dimensions_scores = [m["dimensions"] for m in metrics]

            norm_sharpness = normalize_scores(sharpness_scores)
            norm_confidence = normalize_scores(confidence_scores)
            norm_dimensions = normalize_scores(dimensions_scores)

            # Calculate composite scores
            best_idx = -1
            best_composite_score = -1

            for idx, metric in enumerate(metrics):
                composite_score = calculate_composite_score(
                    norm_sharpness[idx],
                    norm_confidence[idx],
                    norm_dimensions[idx],
                    sharpness_weight,
                    confidence_weight,
                    dimensions_weight,
                )

                detection = metric["detection"]
                print(
                    f"  Frame {detection.frame_number}: "
                    f"sharpness={metric['sharpness']:.2f} (norm={norm_sharpness[idx]:.2f}), "
                    f"conf={metric['confidence']:.2f} (norm={norm_confidence[idx]:.2f}), "
                    f"dim={metric['dimensions']} (norm={norm_dimensions[idx]:.2f}), "
                    f"composite={composite_score:.3f}"
                )

                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_idx = idx

            # Save the best image for this track
            if best_idx >= 0:
                best_metric = metrics[best_idx]
                best_detection = best_metric["detection"]
                best_img = best_metric["img"]

                # Convert image to bytes
                _, buffer = cv2.imencode(".jpg", best_img)
                img_bytes = buffer.tobytes()

                # Create Track record
                track_record = CombinedDetection(
                    track_id=track_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    avg_conf=avg_conf,
                    unique_labels=unique_labels_str,
                    label=majority_label,
                    img_data=img_bytes,
                )

                # Save to database (update if exists)
                existing_track = session.exec(
                    select(CombinedDetection).where(
                        CombinedDetection.track_id == track_id
                    )
                ).first()

                if existing_track:
                    # Update existing record
                    existing_track.start_frame = start_frame
                    existing_track.end_frame = end_frame
                    existing_track.avg_conf = avg_conf
                    existing_track.unique_labels = unique_labels_str
                    existing_track.label = majority_label
                    existing_track.img_data = img_bytes
                else:
                    # Add new record
                    session.add(track_record)

                session.commit()

                # Also save to file
                label_str = sanitize_filename(majority_label)
                filename = f"track_{track_id}_{label_str}.jpg"
                filepath = output_dir / filename

                success = cv2.imwrite(str(filepath), best_img)

                if success:
                    saved_count += 1
                    print(f"  ✓ Saved to database and file: {filename}")
                    print(
                        f"    Track: frames {start_frame}-{end_frame}, "
                        f"avg_conf={avg_conf:.2f}, label={majority_label}"
                    )
                    print(
                        f"    Best frame: {best_detection.frame_number}, "
                        f"composite score: {best_composite_score:.3f}"
                    )
                else:
                    print(f"  ✗ Failed to save file: {filename}")
            else:
                print(f"  ✗ No valid images found for track_id {track_id}")

        print(f"\n{'=' * 60}")
        print(f"Processing complete!")
        print(f"Saved {saved_count}/{len(tracks)} best images")
        print(f"Database: Track table updated with {saved_count} records")
        print(f"Files: {output_dir.absolute()}")


if __name__ == "__main__":
    # You can adjust the weights here
    # For example, to prioritize sharpness more:
    # select_best_images(sharpness_weight=0.5, confidence_weight=0.25, dimensions_weight=0.25)

    select_best_images()
