#!/usr/bin/env zsh

# ./script/reset.sh
# uv run main_img.py --no-draw
# uv run duplicates.py
# cp -r out app/out

rm -rf out
rm detections.db
uv run yolo_db.py
uv run get_best.py
