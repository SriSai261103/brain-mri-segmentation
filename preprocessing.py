import cv2
import numpy as np
from skimage import exposure

# Apply CLAHE to enhance MRI images
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    if len(image.shape) == 3:  # If image has multiple channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image_clahe = clahe.apply(image)
    return image_clahe

# Apply normalization and additional preprocessing steps
def preprocess_image(image):
    image = apply_clahe(image)
    image = exposure.rescale_intensity(image, out_range=(0, 1))
    # Further augmentation steps like rotations, flips can be added here
    return image
