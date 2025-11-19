from collections import defaultdict
from typing import List, Set, Tuple

from sqlmodel import Session, select

from src.database import Detection


def calculate_iou(
    bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def detect_cyclists(
    session: Session, iou_threshold: float = 0.1, min_consecutive_frames: int = 3
) -> List[dict]:
    """
    Detect cyclists by finding overlapping person and bike bboxes.

    Args:
        session: SQLModel database session
        iou_threshold: Minimum IoU to consider overlap (0.1 = 10% overlap)
        min_consecutive_frames: Minimum frames of overlap to confirm cyclist

    Returns:
        List of cyclist detections with frame ranges
    """
    # Get all person and bike detections, ordered by frame
    statement = (
        select(Detection)
        .where((Detection.label == "person") | (Detection.label == "bike"))
        .order_by(Detection.frame_number)
    )

    detections = session.exec(statement).all()

    # Group detections by frame
    frames = defaultdict(lambda: {"person": [], "bike": []})
    for det in detections:
        frames[det.frame_number][det.label].append(det)

    # Track potential cyclists across frames
    # Key: (person_track_id, bike_track_id), Value: list of frame_numbers
    cyclist_tracks = defaultdict(list)

    # Find overlapping person-bike pairs in each frame
    for frame_num in sorted(frames.keys()):
        persons = frames[frame_num]["person"]
        bikes = frames[frame_num]["bike"]

        for person in persons:
            person_bbox = (
                person.bbox_x1,
                person.bbox_y1,
                person.bbox_x2,
                person.bbox_y2,
            )

            for bike in bikes:
                bike_bbox = (bike.bbox_x1, bike.bbox_y1, bike.bbox_x2, bike.bbox_y2)

                iou = calculate_iou(person_bbox, bike_bbox)

                if iou >= iou_threshold:
                    pair_id = (person.track_id, bike.track_id)
                    cyclist_tracks[pair_id].append(frame_num)

    # Filter for cyclists with sufficient consecutive frames
    cyclists = []
    for (person_id, bike_id), frame_list in cyclist_tracks.items():
        frame_list.sort()

        # Find consecutive sequences
        sequences = []
        current_seq = [frame_list[0]]

        for i in range(1, len(frame_list)):
            if frame_list[i] - frame_list[i - 1] <= 2:  # Allow 1 frame gap
                current_seq.append(frame_list[i])
            else:
                if len(current_seq) >= min_consecutive_frames:
                    sequences.append(current_seq)
                current_seq = [frame_list[i]]

        if len(current_seq) >= min_consecutive_frames:
            sequences.append(current_seq)

        # Add confirmed cyclist detections
        for seq in sequences:
            cyclists.append(
                {
                    "person_track_id": person_id,
                    "bike_track_id": bike_id,
                    "start_frame": seq[0],
                    "end_frame": seq[-1],
                    "num_frames": len(seq),
                    "frames": seq,
                }
            )

    return cyclists


# Usage example
def main():
    from sqlmodel import create_engine

    engine = create_engine("sqlite:///your_database.db")

    with Session(engine) as session:
        cyclists = detect_cyclists(
            session,
            iou_threshold=0.1,  # 10% overlap
            min_consecutive_frames=5,  # At least 5 frames
        )

        print(f"Found {len(cyclists)} cyclist detections:")
        for c in cyclists:
            print(
                f"  Person {c['person_track_id']} + Bike {c['bike_track_id']}: "
                f"Frames {c['start_frame']}-{c['end_frame']} ({c['num_frames']} frames)"
            )


if __name__ == "__main__":
    main()
