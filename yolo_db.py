import argparse

import cv2
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from ultralytics import YOLO

from src.database import init_db, save_detections

# --- CONFIG ---
MODEL_PATH = "models/yolo11m.pt"
VIDEO_PATH = "video/glen-oliver/short/before_glen-oliver.mp4"
MIN_CONFIDENCE = 0.6  # only consider detections above this confidence
CROP_PADDING = 10  # pixels of padding around the bbox
PROCESS_EVERY_N_FRAMES = 3  # Process every nth frame
# ----------------

# Parse command-line arguments
parser = argparse.ArgumentParser(description="YOLO11 object tracking")
parser.add_argument(
    "--no-draw",
    action="store_true",
    help="Disable display window and frame visualization",
)
args = parser.parse_args()

# Initialize database
init_db()

# Load the YOLO11 model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames_to_process = total_frames // PROCESS_EVERY_N_FRAMES

frame_number = 0

# Create progress bar only if no-draw is enabled
if args.no_draw:
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total} frames"),
        TimeRemainingColumn(),
    )
    progress.start()
    task = progress.add_task("[cyan]Processing video...", total=frames_to_process)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    verbose = True
    if args.no_draw:
        verbose = False

    if success:
        # check if we should process this frame
        if frame_number % PROCESS_EVERY_N_FRAMES != 0:
            frame_number += 1
            continue

        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(
            frame, persist=True, tracker="bytetrack_config.yaml", verbose=verbose
        )

        # Visualize and display only if drawing is enabled
        if not args.no_draw:
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Update progress bar
            progress.update(task, advance=1)

        # Save detections to database
        save_detections(results, frame, frame_number, MIN_CONFIDENCE, CROP_PADDING)

        frame_number += 1
    else:
        # Break the loop if the end of the video is reached
        break

# Stop progress bar if it was started
if args.no_draw:
    progress.stop()

# Release the video capture object and close the display window
cap.release()
if not args.no_draw:
    cv2.destroyAllWindows()

print(f"Processing complete. Total frames processed: {frame_number}")
