#!/usr/bin/env zsh

./script/reset.sh

uv run main_img.py --no-draw
uv run duplicates.py

cp -r out app/out

