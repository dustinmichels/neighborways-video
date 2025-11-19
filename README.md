# Neighborways Video

New stuff:

```sh
rm -rf out
rm detections.db
uv run yolo_db.py
uv run export.py
```

Also:

```sh
open -a "DB Browser for SQLite" detections.db
```

To run:

```sh
# with video
uv run main_img.py

# without video
uv run main_img.py --no-draw

# check for duplicates
uv run duplicates.py

# copy output
cp -r out app/out

```

Run notebooks:

```sh
uv run --with jupyter jupyter lab
```

Clear video:

```sh
uv run --with jupyter ./clear_notebooks.sh
```

Make samples of videos:

```sh
./make_short_videos.sh
```

## About video

| Date      | Intersection    | Start Time | End Time |
| --------- | --------------- | ---------- | -------- |
| 6-25-2024 | Broadway & Glen | 5:10 PM    | 5:40 PM  |
| 7-16-2025 | Glen & Broadway | 4:30 PM    | 5:00 PM  |
| 6-25-2024 | Glen & Oliver   | 4:30 PM    | 5:00 PM  |
| 7-16-2025 | Glen & Oliver   | 4:00 PM    | 4:30 PM  |

### My counts

`video/glen-oliver/short/before_glen-oliver.mp4`

People - 1
Bikes - 1
Cars - 7
-> Van - 2
