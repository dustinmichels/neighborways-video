import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set

import cv2
import numpy as np
from rich.console import Console
from rich.progress import (MofNCompleteColumn, Progress, SpinnerColumn,
                           TimeElapsedColumn)
from rich.table import Table
from ultralytics import YOLO

from src.models import ImgRecord

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="YOLO object tracking and crop saving")
parser.add_argument(
    "--no-draw",
    action="store_true",
    help="Disable drawing boxes and displaying video feed (faster processing)",
)
args = parser.parse_args()

# --- CONFIG ---
# MODEL_PATH = "yolov8s.pt"
# MODEL_PATH = "yolov10m.pt"
MODEL_PATH = "yolo11l.pt"
# VIDEO_PATH = "video/glen-oliver/after_glen-oliver.mp4"
VIDEO_PATH = "video/glen-oliver/short/before_glen-oliver.mp4"
OUTPUT_DIR = Path("out/saved_unique_crops")
IMG_DIR = OUTPUT_DIR / "img"
MIN_CONFIDENCE = 0.5  # only consider detections above this confidence
CROP_PADDING = 10  # pixels of padding around the bbox
CROP_SIZE = (256, 256)  # saved crop size (width, height), or None to keep original
ENABLE_DRAWING = not args.no_draw  # Control drawing based on command line arg
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
STATS_UPDATE_INTERVAL = 30  # Update console stats every N frames
# ----------------

# Delete output dir if it exists
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# List to store ImgRecord instances
manifest_records: List[ImgRecord] = []

# Initialize console for rich output
console = Console()

# Load YOLO model with verbose=False to suppress output
console.print("[bold cyan]Loading YOLO model...[/bold cyan]")
model = YOLO(MODEL_PATH, verbose=False)
console.print(f"[green]✓[/green] Model loaded: {MODEL_PATH}\n")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

console.print(f"[bold]Video Information:[/bold]")
console.print(f"  Resolution: {width}x{height}")
console.print(f"  FPS: {fps}")
console.print(f"  Total Frames: {total_frames}")
console.print(f"  Processing every {PROCESS_EVERY_N_FRAMES} frames\n")

# Track seen IDs per class
unique_objects: Dict[str, Set[int]] = {
    "person": set(),
    "bicycle": set(),
    "car": set(),
    "motorbike": set(),
    "bus": set(),
    "truck": set(),
}

# Track current frame detections
current_detections: Dict[str, int] = {
    "person": 0,
    "bicycle": 0,
    "car": 0,
    "motorbike": 0,
    "bus": 0,
    "truck": 0,
}

# Colors for on-screen drawing
colors = {
    "person": (0, 255, 0),  # Bright Green
    "bicycle": (255, 0, 255),  # Magenta
    "car": (255, 0, 0),  # Red
    "motorbike": (255, 165, 0),  # Orange
    "bus": (0, 255, 255),  # Cyan
    "truck": (128, 0, 255),  # Purple
}

frame_no = 0
saved_this_session = 0


def draw_stats_panel(frame, unique_counts, current_counts, frame_no, total_saved):
    """Draw a statistics panel on the frame"""
    panel_height = 240
    panel_width = 320
    panel_x = 10
    panel_y = 10
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "TRACKING STATISTICS", 
                (panel_x + 10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Frame info
    cv2.putText(frame, f"Frame: {frame_no}", 
                (panel_x + 10, panel_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, f"Total Saved: {total_saved}", 
                (panel_x + 10, panel_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Column headers
    y_offset = panel_y + 100
    cv2.putText(frame, "Class", (panel_x + 10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, "Unique", (panel_x + 120, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, "Current", (panel_x + 220, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Draw separator line
    y_offset += 5
    cv2.line(frame, (panel_x + 10, y_offset), 
             (panel_x + panel_width - 10, y_offset), (100, 100, 100), 1)
    
    # Stats for each class
    y_offset += 15
    for class_name in unique_objects.keys():
        color = colors.get(class_name, (255, 255, 255))
        unique_count = len(unique_counts[class_name])
        current_count = current_counts[class_name]
        
        # Class name with color indicator
        cv2.circle(frame, (panel_x + 20, y_offset - 3), 4, color, -1)
        cv2.putText(frame, class_name.capitalize(), 
                    (panel_x + 30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Unique count
        cv2.putText(frame, str(unique_count), 
                    (panel_x + 135, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Current count
        cv2.putText(frame, str(current_count), 
                    (panel_x + 235, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
        
        y_offset += 20
    
    return frame


def print_stats_table(unique_counts, current_counts, frame_no, total_saved):
    """Print statistics table to console"""
    table = Table(title=f"Tracking Statistics (Frame {frame_no})")
    
    table.add_column("Class", style="cyan", no_wrap=True)
    table.add_column("Unique Objects", style="green", justify="right")
    table.add_column("Currently Visible", style="yellow", justify="right")
    
    for class_name in unique_objects.keys():
        unique_count = len(unique_counts[class_name])
        current_count = current_counts[class_name]
        table.add_row(
            class_name.capitalize(),
            str(unique_count),
            str(current_count)
        )
    
    table.add_row("", "", "", style="dim")
    table.add_row("[bold]TOTAL SAVED[/bold]", f"[bold green]{total_saved}[/bold green]", "", style="bold")
    
    console.print(table)
    console.print()


# Create progress bar (only when not drawing to screen)
if not ENABLE_DRAWING:
    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    progress.start()
    task = progress.add_task("[cyan]Processing frames...", total=total_frames)

# Print initial stats
console.print("[bold green]Starting video processing...[/bold green]\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    # Skip frames for faster processing
    if frame_no % PROCESS_EVERY_N_FRAMES != 0:
        if not ENABLE_DRAWING:
            progress.update(task, advance=1)  # Still update progress bar
        continue

    # Reset current frame detection counts
    for key in current_detections:
        current_detections[key] = 0

    # Run tracking with verbose=False to suppress output
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    for r in results:

        if not hasattr(r, "boxes") or r.boxes is None:
            continue

        boxes = r.boxes
        for i in range(len(boxes)):
            box = boxes[i]
            try:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = (
                    int(box.id[0]) if getattr(box, "id", None) is not None else None
                )
                xyxy = (
                    box.xyxy[0].cpu().numpy()
                    if hasattr(box.xyxy[0], "cpu")
                    else np.array(box.xyxy[0])
                )
            except Exception:
                continue

            if conf < MIN_CONFIDENCE:
                continue

            label = model.names.get(cls_id, str(cls_id))
            if label not in unique_objects:
                continue

            if track_id is None:
                continue

            # Count current detections
            current_detections[label] += 1

            # Save only the first appearance of each unique track ID per class
            if track_id not in unique_objects[label]:
                unique_objects[label].add(track_id)
                saved_this_session += 1

                x1, y1, x2, y2 = map(int, xyxy)
                h, w = frame.shape[:2]

                # Ensure the entire bounding box (plus padding) is captured safely
                x1p = max(0, min(x1 - CROP_PADDING, w - 1))
                y1p = max(0, min(y1 - CROP_PADDING, h - 1))
                x2p = max(0, min(x2 + CROP_PADDING, w))
                y2p = max(0, min(y2 + CROP_PADDING, h))

                # Defensive check
                if x2p <= x1p or y2p <= y1p:
                    continue

                crop = frame[y1p:y2p, x1p:x2p].copy()
                if crop.size == 0:
                    continue

                # Optionally resize the crop
                if CROP_SIZE is not None:
                    crop = cv2.resize(crop, CROP_SIZE)

                # Generate filename with frame number first, then label
                filename = f"f{frame_no:06d}_{label}_id{track_id}_c{conf:.2f}.jpg"
                save_path = IMG_DIR / filename

                # Save the cropped image
                cv2.imwrite(str(save_path), crop)

                # Create ImgRecord and add to manifest
                record = ImgRecord(
                    saved_path=str(save_path),
                    label=label,
                    track_id=track_id,
                    frame_no=frame_no,
                    conf=conf,
                )
                manifest_records.append(record)

                # Visual notification (only if drawing enabled)
                if ENABLE_DRAWING:
                    cv2.putText(
                        frame,
                        f"NEW: {label.upper()} #{track_id}",
                        (x1, max(15, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

            # Draw bbox on display frame (only if drawing enabled)
            if ENABLE_DRAWING:
                color = colors.get(label, (0, 255, 0))
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} #{track_id} {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

    # Print stats to console periodically (when not drawing)
    if not ENABLE_DRAWING and frame_no % STATS_UPDATE_INTERVAL == 0:
        print_stats_table(unique_objects, current_detections, frame_no, saved_this_session)

    # Update display or progress bar
    if ENABLE_DRAWING:
        # Draw statistics panel on frame
        frame = draw_stats_panel(frame, unique_objects, current_detections, 
                                 frame_no, saved_this_session)
        
        cv2.imshow("Object Tracking with Statistics", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Update progress bar
        progress.update(task, advance=1)

cap.release()
if ENABLE_DRAWING:
    cv2.destroyAllWindows()
else:
    progress.stop()

# Print final statistics
console.print("\n[bold green]Processing Complete![/bold green]\n")
print_stats_table(unique_objects, current_detections, frame_no, saved_this_session)

console.print(f"[bold]Output Directory:[/bold] {OUTPUT_DIR.resolve()}")
console.print(f"[bold]Images Directory:[/bold] {IMG_DIR.resolve()}\n")

# Write CSV manifest from ImgRecord objects
console.print("[cyan]Writing CSV manifest...[/cyan]")
manifest_csv_path = OUTPUT_DIR / "manifest.csv"
with open(manifest_csv_path, "w", newline="") as csvfile:
    if manifest_records:
        # Use the field names from the Pydantic model
        fieldnames = list(ImgRecord.model_fields.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in manifest_records:
            writer.writerow(record.model_dump())

console.print(f"[green]✓[/green] CSV manifest: {manifest_csv_path.resolve()}")

# Write JSON manifest from ImgRecord objects
console.print("[cyan]Writing JSON manifest...[/cyan]")
json_manifest_path = OUTPUT_DIR / "manifest.json"
with open(json_manifest_path, "w") as jsonfile:
    # Convert all records to dictionaries
    manifest_data = [record.model_dump() for record in manifest_records]
    json.dump(manifest_data, jsonfile, indent=2)

console.print(f"[green]✓[/green] JSON manifest: {json_manifest_path.resolve()}")

# Print summary
console.print("\n[bold green]═══ FINAL SUMMARY ═══[/bold green]")
summary_table = Table(show_header=False, box=None)
summary_table.add_column(style="cyan bold")
summary_table.add_column(style="white")

summary_table.add_row("Total Frames Processed:", str(frame_no))
summary_table.add_row("Unique Objects Tracked:", str(saved_this_session))
summary_table.add_row("Image Crops Saved:", str(len(manifest_records)))

for class_name in sorted(unique_objects.keys()):
    count = len(unique_objects[class_name])
    if count > 0:
        summary_table.add_row(f"  {class_name.capitalize()}:", str(count))

console.print(summary_table)
console.print()