# remove existing short videos
rm -rf video/glen-oliver/short
rm -rf video/glen-broadway/short

# Create destination folders
mkdir -p video/glen-oliver/short
mkdir -p video/glen-broadway/short

# Loop over both locations
for place in glen-oliver glen-broadway; do
  for part in before after; do
    ffmpeg -i "video/${place}/${part}_${place}.mp4" \
    -ss 300 -t 45 -c copy "video/${place}/short/${part}_${place}.mp4"
  done
done

