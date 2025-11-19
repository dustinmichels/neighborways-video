# TODO

For each object detected, I want to save a record with:
id, label, confidence, bounding box, frame, img
(the actual img)
to a sqlite database

Save every frame of an object being tracked. Then use a model to chose the best one.

- Post-processing: If bike and pedestrian bounding boxes are overlapping, call it "cyclist"
