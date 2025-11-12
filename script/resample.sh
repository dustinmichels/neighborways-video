#!/bin/zsh

# remove existing resampled videos
rm -rf video/glen-oliver/short-resample
rm -rf video/glen-broadway/short-resample

# create new destination folders
mkdir -p video/glen-oliver/short-resample
mkdir -p video/glen-broadway/short-resample

max_jobs=4
current_jobs=0

# loop over locations and short videos
for place in glen-oliver glen-broadway; do
  for part in before after; do
    input="video/${place}/short/${part}_${place}.mp4"
    output="video/${place}/short-resample/${part}_${place}.mp4"

    echo "Resampling $input to 15 FPS"

    # run ffmpeg in background with fixed 15 FPS
    ffmpeg -y -i "$input" -filter:v "fps=15" -c:a copy "$output" &

    # limit concurrent jobs
    ((current_jobs++))
    if (( current_jobs >= max_jobs )); then
      wait
      current_jobs=0
    fi
  done
done

# wait for any remaining background jobs
wait
echo "Resampling complete!"
