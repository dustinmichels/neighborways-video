for nb in notebooks/*.ipynb; do
  jupyter nbconvert --clear-output --inplace "$nb"
done
