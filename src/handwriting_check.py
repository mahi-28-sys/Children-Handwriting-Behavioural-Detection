import os
import cv2
import numpy as np

def _filter_page_lines(lines, img_shape, long_frac=0.55, border_frac=0.10, angle_tol_deg=10):
    """Filter out long page border or header lines (not handwriting)."""
    if lines is None:
        return []

    h, w = img_shape[:2]
    max_side = max(h, w)
    filtered = []
    for l in lines:
        coords = np.ravel(l)  # handles both (N,1,4) and (N,4) shaped output
        if coords.size < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2) ** 0.5
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        angle = min(angle, 180 - angle)  # Map to [0,90]

        near_top = (y1 < border_frac*h and y2 < border_frac*h)
        near_bottom = (y1 > (1-border_frac)*h and y2 > (1-border_frac)*h)
        near_left = (x1 < border_frac*w and x2 < border_frac*w)
        near_right = (x1 > (1-border_frac)*w and x2 > (1-border_frac)*w)

        is_long = length > long_frac * max_side
        is_horiz = angle <= angle_tol_deg
        is_vert = angle >= (90 - angle_tol_deg)

        # Skip page-border or long header lines
        if is_long and (is_horiz or is_vert) and (near_top or near_bottom or near_left or near_right):
            continue
        if is_long and is_horiz and (y1 < 0.2*h):
            continue

        filtered.append((x1, y1, x2, y2, length, angle))
    return filtered


def _crop_to_content(img, margin_frac=0.02, min_ink_frac=0.0005):
    """
    Crop a grayscale image down to the bounding box of its dark content
    (text/ink), so metrics like stroke density aren't diluted by large
    blank margins around a small block of handwriting.
    Falls back to the original image if no clear content is found.
    """
    h, w = img.shape[:2]

    # Simple global threshold is enough just to find the content region
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ink_frac = np.count_nonzero(mask) / mask.size
    if ink_frac < min_ink_frac:
        # Essentially nothing dark on the page at all — nothing to crop to
        return img

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img

    y1, y2 = np.argmax(rows), h - np.argmax(rows[::-1])
    x1, x2 = np.argmax(cols), w - np.argmax(cols[::-1])

    margin_y = int(margin_frac * h)
    margin_x = int(margin_frac * w)
    y1 = max(0, y1 - margin_y)
    y2 = min(h, y2 + margin_y)
    x1 = max(0, x1 - margin_x)
    x2 = min(w, x2 + margin_x)

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img
    return cropped


def _detect_faces(img):
    """Detect faces in the image using Haar Cascade."""
    try:
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        if face_cascade.empty():
            return 0
        
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return len(faces)
    except:
        return 0


def _detect_color_photo(img_path):
    """Detect if image is a color photograph."""
    try:
        img_color = cv2.imread(img_path)
        if img_color is None:
            return False, {}
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        
        # Calculate saturation statistics
        saturation = hsv[:, :, 1]
        mean_saturation = np.mean(saturation)
        high_saturation_ratio = np.sum(saturation > 50) / saturation.size
        
        # Calculate color variance
        b, g, r = cv2.split(img_color)
        color_variance = np.mean([np.var(b), np.var(g), np.var(r)])
        
        # Check if channels are different (color vs grayscale)
        channel_diff = np.mean(np.abs(b.astype(float) - g.astype(float))) + \
                       np.mean(np.abs(g.astype(float) - r.astype(float)))
        
        is_color = (mean_saturation > 25 or 
                   high_saturation_ratio > 0.15 or 
                   channel_diff > 10)
        
        return is_color, {
            "mean_saturation": float(mean_saturation),
            "high_saturation_ratio": float(high_saturation_ratio),
            "color_variance": float(color_variance),
            "channel_diff": float(channel_diff)
        }
    except:
        return False, {}


def _detect_photo_characteristics(img, th):
    """Detect characteristics that indicate a photograph rather than handwriting."""
    
    # Check for high variation in intensity (photos have more gradients)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    laplacian_var = cv2.Laplacian(blur, cv2.CV_64F).var()
    
    # Calculate histogram to detect photo-like distributions
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Photos typically have more varied intensity distribution
    mid_range_intensity = np.sum(hist[50:200])
    
    # Check for color complexity in grayscale distribution
    hist_peaks = np.sum(hist > 0.01)
    
    # Calculate standard deviation of pixel intensities
    std_dev = np.std(img)
    
    return {
        "laplacian_variance": float(laplacian_var),
        "mid_range_intensity": float(mid_range_intensity),
        "intensity_peaks": int(hist_peaks),
        "std_dev": float(std_dev)
    }


def is_handwriting_image(img_path, debug=False):
    """Detect whether an image is likely a handwriting sample, including scanned pages."""

    if not os.path.exists(img_path):
        return (False, {"error": "File not found"}) if debug else False

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (False, {"error": "Cannot read image"}) if debug else False

    h, w = img.shape[:2]
    
    if max(h, w) > 1800:
        scale = 1800.0 / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        h, w = img.shape[:2]
    else:
        scale = 1.0

    # Crop down to the content region so a small handwriting block on a
    # mostly-blank page doesn't get diluted stroke/edge/contour metrics.
    img = _crop_to_content(img)
    h, w = img.shape[:2]

    # Detect if it's a color photo
    is_color, color_metrics = _detect_color_photo(img_path)
    
    # Detect faces
    face_count = _detect_faces(img)
    
    # Adaptive threshold for scanned images
    th = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 15
    )

    # Detect photo characteristics
    photo_chars = _detect_photo_characteristics(img, th)
    
    # Contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    contour_areas = [cv2.contourArea(c) for c in contours]
    avg_contour_area = float(np.mean(contour_areas)) if contour_areas else 0
    median_contour_area = float(np.median(contour_areas)) if contour_areas else 0
    
    # Calculate large contour ratio (photos often have few large regions)
    large_contours = sum(1 for area in contour_areas if area > 1000)
    large_contour_ratio = large_contours / max(contour_count, 1)

    # Edge detection
    edges = cv2.Canny(img, 50, 150)
    edge_density = float(np.sum(edges > 0)) / edges.size

    # Lines
    lines_p = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                              minLineLength=int(0.1*max(h, w)),
                              maxLineGap=int(0.02*max(h, w)))
    filtered_lines = _filter_page_lines(lines_p, img.shape)
    structured_line_count = len(filtered_lines)
    orig_line_count = 0 if lines_p is None else lines_p.shape[0]

    # Stroke density
    ink_pixels = cv2.countNonZero(th)
    stroke_density = ink_pixels / th.size

    # --- Enhanced detection logic ---
    rejection_reason = None
    
    # Reject if faces detected
    if face_count > 0:
        is_hw = False
        rejection_reason = "Personal photo detected (faces found)"
    # Reject if color photograph/screenshot detected
    elif is_color:
        is_hw = False
        rejection_reason = "Color image detected (photo, screenshot, or UI capture)"
    # Reject if photo characteristics detected (high texture complexity)
    elif photo_chars["laplacian_variance"] > 800 and photo_chars["std_dev"] > 60:
        is_hw = False
        rejection_reason = "Photograph detected (complex texture/gradients)"
    # Reject if too many large contours (typical of photos)
    elif large_contour_ratio > 0.3 and avg_contour_area > 3000:
        is_hw = False
        rejection_reason = "Large objects detected (not handwriting)"
    # Reject if stroke density is too high (dark photos) or too low (blank/light photos)
    elif stroke_density > 0.65:
        is_hw = False
        rejection_reason = f"Image too dark (stroke density: {stroke_density:.3f})"
    elif stroke_density < 0.005:
        is_hw = False
        rejection_reason = f"Image too light/blank (stroke density: {stroke_density:.3f})"
    # Reject if edge density too high (complex natural scenes)
    elif edge_density > 0.15:
        is_hw = False
        rejection_reason = "Too complex (natural scene/photograph)"
    # Reject if too many contours (noise/photo)
    elif contour_count > 2000:
        is_hw = False
        rejection_reason = "Too noisy or complex (not handwriting)"
    # Check all handwriting criteria
    elif (edge_density > 0.0015 and
          contour_count > 15 and
          avg_contour_area < 6000 and
          structured_line_count < 400):
        is_hw = True
    else:
        is_hw = False
        rejection_reason = "Does not match handwriting patterns"

    metrics = {
        "stroke_density": round(stroke_density, 5),
        "edge_density": round(edge_density, 5),
        "contour_count": contour_count,
        "avg_contour_area": round(avg_contour_area, 2),
        "median_contour_area": round(median_contour_area, 2),
        "large_contour_ratio": round(large_contour_ratio, 3),
        "orig_line_count": orig_line_count,
        "structured_line_count": structured_line_count,
        "face_count": face_count,
        "laplacian_variance": round(photo_chars["laplacian_variance"], 2),
        "mid_range_intensity": round(photo_chars["mid_range_intensity"], 3),
        "std_dev": round(photo_chars["std_dev"], 2),
        "is_color_photo": is_color,
        **{f"color_{k}": round(v, 3) if isinstance(v, float) else v for k, v in color_metrics.items()},
        "scale": round(scale, 3),
    }

    if debug:
        if is_hw:
            print("✅ Handwriting detected")
            print("📊 Metrics:", metrics)
            return (True, metrics)
        else:
            print("\n❌ ERROR: Not a handwriting image!")
            print(f"🔍 Reason: {rejection_reason}")
            print("\n⚠️  Please upload only handwritten text samples.")
            print("🚫 The following types are NOT accepted:")
            print("   • Personal photos (selfies, portraits)")
            print("   • Animal photos")
            print("   • Flower/plant photos")
            print("   • Nature photographs")
            print("   • Diagrams and graphs")
            print("   • Charts and illustrations")
            print("   • General photographs\n")
            print("📊 Detection metrics:", metrics)
            return (False, {"error": "Not a handwriting image", "reason": rejection_reason, "metrics": metrics})

    return is_hw


if __name__ == "__main__":
    test = "8.jpg"
    result = is_handwriting_image(test, debug=True)
    
    # Example of how to use the result
    if isinstance(result, tuple):
        is_handwriting, info = result
        if not is_handwriting:
            print("\n⛔ Upload rejected. Please try again with a handwriting sample.")
    else:
        if not result:
            print("\n⛔ Upload rejected. Please try again with a handwriting sample.")