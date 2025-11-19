import cv2
import numpy as np


def crop_img(crop, crop_size):
    crop_h, crop_w = crop.shape[:2]
    target_w, target_h = crop_size

    # Calculate scaling factor to fit within target size
    scale = min(target_w / crop_w, target_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    # Resize maintaining aspect ratio
    resized = cv2.resize(crop, (new_w, new_h))

    # Create a black canvas of target size
    crop = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center the resized image on the canvas
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    crop[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return crop
