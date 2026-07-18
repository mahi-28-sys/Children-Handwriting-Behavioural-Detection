import cv2
import numpy as np
import os

def preprocess_image(img_path, save_path=None, resize_dim=(256, 256), margin=10):
    """
    Preprocess handwriting image:
    - Dynamically crop header/footer based on content
    - Convert to grayscale
    - Apply Gaussian blur
    - Apply adaptive thresholding (binarization)
    - Resize grayscale and binary images

    Args:
        img_path (str): Path to input image
        save_path (str, optional): Path to save preprocessed image (binary)
        resize_dim (tuple, optional): Target size (width, height)
        margin (int, optional): Extra pixels to include around content
    
    Returns:
        gray_resized (np.array): Resized grayscale image
        binary_resized (np.array): Resized thresholded image
    """

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not load {img_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Quick binary to detect content
    _, temp_binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find the rows that contain handwriting
    rows = np.any(temp_binary, axis=1)
    if not np.any(rows):
        # No handwriting detected, fallback to original
        top, bottom = 0, gray.shape[0]
    else:
        top = max(np.argmax(rows) - margin, 0)
        bottom = min(gray.shape[0] - np.argmax(rows[::-1]) + margin, gray.shape[0])

    # Crop image dynamically
    cropped = img[top:bottom, :]

    # ---- GRAYSCALE ----
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # ---- GAUSSIAN BLUR ----
    blur = cv2.GaussianBlur(gray_cropped, (5, 5), 0)

    # ---- ADAPTIVE THRESHOLD ----
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # ---- RESIZE BOTH IMAGES ----
    gray_resized = cv2.resize(gray_cropped, resize_dim)
    binary_resized = cv2.resize(binary, resize_dim)

    # ---- SAVE IF PATH PROVIDED ----
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, binary_resized)

    return gray_resized, binary_resized
