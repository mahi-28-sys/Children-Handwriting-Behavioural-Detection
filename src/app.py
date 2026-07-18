from flask import Flask, render_template, request, redirect, url_for, session
import os
import cv2

from handwriting_check import is_handwriting_image
from preprocess import preprocess_image
from predict import (
    predict_image,
    get_combined_suggestions,
    load_models,
)


# Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit
# Secret key is required for Flask's session (used to pass error/result
# data across the redirect-after-POST below). Fine as a fixed string for
# local/personal use; for a real deployment, load this from an environment
# variable instead.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")


# Load models

models = load_models()


# Helpers

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image_keep_aspect(path, max_side=800):
    img = cv2.imread(path)
    if img is None:
        print("⚠ Failed to read image for resizing:", path)
        return path
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        cv2.imwrite(path, img)
        print(f"✅ Image resized to {img.shape[1]}x{img.shape[0]}")
    return path


# Routes

@app.route("/", methods=["GET"])
def index():
    # GET-only route: just shows the upload form.
    # Any error from a previous submission is read from the session
    # (set by the /analyze route below) and immediately cleared, so
    # refreshing this page (F5) never re-shows a stale error message.
    error = session.pop("error", None)
    return render_template("index.html", error=error)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("handwriting_file")
    if not file or file.filename == "":
        session["error"] = "Please choose an image."
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        session["error"] = "Only .png, .jpg, .jpeg allowed."
        return redirect(url_for("index"))

    # secure_filename requires werkzeug.utils - imported below to
    # keep this file's top-level imports minimal and explicit
    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    print("✅ File saved at:", filepath)

    # Step 1: Resize
    filepath = resize_image_keep_aspect(filepath)

    # Step 2: Preprocess
    try:
        gray, binary = preprocess_image(filepath)
        print("✅ Preprocessing done")
    except Exception as e:
        print("⚠ Preprocessing failed:", e)
        gray, binary = None, None

    # Step 3: Handwriting detection
    try:
        if os.environ.get("HW_DEBUG") == "1":
            is_hw, metrics = is_handwriting_image(filepath, debug=True)
            print("📝 Handwriting metrics:", metrics)
        else:
            is_hw = is_handwriting_image(filepath)
    except Exception as e:
        print("⚠ Handwriting check error:", e)
        is_hw = False

    if not is_hw:
        session["error"] = (
            "❌ This image doesn't appear to contain clear handwriting. "
            "Please upload a scanned or photo image of actual handwritten text "
            "(not printed text, drawings, or blank pages)."
        )
        return redirect(url_for("index"))

    # Step 4: Predict traits (shared logic with predict.py, no duplication)
    results, features = predict_image(models, filepath, return_features=True)
    advice_paragraph = get_combined_suggestions(results)

    return render_template(
        "result.html",
        predictions=results,
        advice_paragraph=advice_paragraph,
        features=features,
    )


if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    # Debug mode is off by default. Set FLASK_DEBUG=1 in your environment
    # only for local development - never enable it in production.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")