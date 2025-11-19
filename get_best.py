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
    For each track_id, select and save the highest quality image based on composite score.

    Args:
        sharpness_weight: Weight for sharpness metric (default: 0.4)
        confidence_weight: Weight for confidence metric (default: 0.3)
        dimensions_weight: Weight for dimensions metric (default: 0.3)
    """
    # Create output directory
    output_dir = Path("out/best")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selecting best images and saving to: {output_dir.absolute()}")
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
            images = []

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
                images.append(img)

            if not metrics:
                print(f"  ✗ No valid images found for track_id {track_id}")
                continue

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

                label = sanitize_filename(best_detection.label)
                filename = f"track_{track_id}_{label}.jpg"
                filepath = output_dir / filename

                success = cv2.imwrite(str(filepath), best_img)

                if success:
                    saved_count += 1
                    print(f"  ✓ Saved best image: {filename}")
                    print(
                        f"    (Frame {best_detection.frame_number}, "
                        f"composite score: {best_composite_score:.3f})"
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
    # You can adjust the weights here
    # For example, to prioritize sharpness more:
    # select_best_images(sharpness_weight=0.5, confidence_weight=0.25, dimensions_weight=0.25)

    select_best_images()
