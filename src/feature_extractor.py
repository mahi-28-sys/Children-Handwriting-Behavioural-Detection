import os
import cv2
import numpy as np
import pandas as pd
from preprocess import preprocess_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "features.csv")
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")

def _proj_stats(bin_img):
    row_sum = bin_img.sum(axis=1) / 255.0
    col_sum = bin_img.sum(axis=0) / 255.0
    r_mean = float(np.mean(row_sum) / bin_img.shape[1])
    r_std  = float(np.std(row_sum) / bin_img.shape[1])
    c_mean = float(np.mean(col_sum) / bin_img.shape[0])
    c_std  = float(np.std(col_sum) / bin_img.shape[0])
    return r_mean, r_std, c_mean, c_std

def _edge_density(gray):
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges) / edges.size)

def _slant_estimate(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) + 1e-6
    ang = np.degrees(np.arctan2(gy, gx))
    mask = mag > np.percentile(mag, 60)
    if not np.any(mask):
        return 0.0
    return float(np.mean(ang[mask]) / 90.0)

def _connected_components(bin_img):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    if num_labels <= 1:
        return 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float(len(areas) / 400.0), float(np.mean(areas) / (bin_img.size))

def extract_handwriting_features(image_path: str) -> dict:
    gray, bin_img = preprocess_image(image_path)

    ink_density = float(np.count_nonzero(bin_img) / bin_img.size)
    mean_intensity = float(np.mean(gray) / 255.0)
    std_intensity = float(np.std(gray) / 255.0)

    r_mean, r_std, c_mean, c_std = _proj_stats(bin_img)
    baseline_wobble = r_std

    edge_density = _edge_density(gray)
    slant = np.clip(_slant_estimate(gray), -1.0, 1.0)
    comp_count_norm, comp_area_norm = _connected_components(bin_img)

    features = {
        "neatness": 1.0 - edge_density,
        "spacing": c_std,
        "slant": slant,
        "letter_size": comp_area_norm,
        "pressure": 1.0 - mean_intensity,
        "baseline_shift": baseline_wobble,
        "loop_size": comp_area_norm,
        "stroke_speed": edge_density,
        "connection_type": comp_count_norm,
        "stroke_variation": std_intensity
    }
    return features

def extract_features_from_folder(dataset_folder: str = None, save_csv=True) -> pd.DataFrame:
    if dataset_folder is None:
        dataset_folder = DEFAULT_DATASET_DIR

    image_paths = []
    for root, _, files in os.walk(dataset_folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print(f"❌ No images found under: {dataset_folder}")
        return pd.DataFrame()

    rows, names = [], []
    for p in image_paths:
        try:
            feats = extract_handwriting_features(p)
            rows.append(feats)
            names.append(os.path.basename(p))
        except Exception as e:
            print(f"⚠ Skipped {p}: {e}")

    df = pd.DataFrame(rows, index=names)
    if save_csv:
        df.to_csv(FEATURES_CSV, index_label="image_name")
        print(f"✅ Saved features.csv with {len(df)} samples and {df.shape[1]} features.")
    return df

if __name__ == "__main__":
    extract_features_from_folder(DEFAULT_DATASET_DIR, save_csv=True)