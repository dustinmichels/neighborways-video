import cv2
from ultralytics import YOLO

from src.database import init_db, save_detections

# --- CONFIG ---
MODEL_PATH = "models/yolo11m.pt"
VIDEO_PATH = "video/glen-oliver/short/before_glen-oliver.mp4"
MIN_CONFIDENCE = 0.6  # only consider detections above this confidence
CROP_PADDING = 10  # pixels of padding around the bbox
PROCESS_EVERY_N_FRAMES = 10  # Process every nth frame
# ----------------


# Initialize database
init_db()

# Load the YOLO11 model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

frame_number = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # check if we should process this frame
        if frame_number % PROCESS_EVERY_N_FRAMES != 0:
            frame_number += 1
            continue

        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="bytetrack_config.yaml")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Save detections to database
        save_detections(results, frame, frame_number, MIN_CONFIDENCE, CROP_PADDING)

        frame_number += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

print(f"Processing complete. Total frames processed: {frame_number}")
