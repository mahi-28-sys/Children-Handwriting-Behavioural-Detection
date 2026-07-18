import os
import joblib
import pandas as pd
from feature_extractor import extract_handwriting_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

trait_features_map = {
    "Academic":        ["neatness", "spacing", "baseline_shift", "letter_size"],
    "Confidence":      ["slant", "letter_size", "pressure", "connection_type"],
    "Creativity":      ["loop_size", "slant", "stroke_variation", "baseline_shift"],
    "Drawing":         ["loop_size", "stroke_variation", "pressure", "letter_size"],
    "Public_Speaking": ["slant", "spacing", "connection_type", "stroke_speed"],
    "Sports":          ["pressure", "stroke_speed", "letter_size"]
}

suggestions_dict = {
    "Academic": {
        "High": "Maintain your study habits and keep challenging yourself",
        "Moderate": "Focus on improving consistency in your studies",
        "Low": "Seek help from teachers to improve academic understanding"
    },
    "Confidence": {
        "High": "Use your confidence to help and inspire others",
        "Moderate": "Participate more in group discussions to build confidence",
        "Low": "Practice public speaking or group activities to boost confidence"
    },
    "Creativity": {
        "High": "Continue exploring innovative ideas",
        "Moderate": "Engage in creative hobbies to develop your skills",
        "Low": "Try activities like drawing or storytelling to spark creativity"
    },
    "Drawing": {
        "High": "Consider showcasing your artwork in school events",
        "Moderate": "Practice regularly to sharpen your skills",
        "Low": "Experiment with basic art techniques to improve drawing"
    },
    "Public_Speaking": {
        "High": "Lead more presentations and debates",
        "Moderate": "Join clubs or events to practice speaking",
        "Low": "Start with small group interactions to build comfort"
    },
    "Sports": {
        "High": "Compete in sports tournaments",
        "Moderate": "Play regularly to enhance your stamina and skills",
        "Low": "Start with basic exercises to improve physical fitness"
    }
}

def load_models():
    models = {}
    for trait in trait_features_map.keys():
        path = os.path.join(MODELS_DIR, f"{trait.lower()}_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {path}. Train first.")
        models[trait] = joblib.load(path)
    return models

def predict_image(models, image_path: str, return_features: bool = False):
    feats = extract_handwriting_features(image_path)
    feats_df = pd.DataFrame([feats])
    out = {}
    for trait, model in models.items():
        cols = trait_features_map[trait]
        pred = model.predict(feats_df[cols])[0]
        out[trait] = pred

    if return_features:
        return out, feats
    return out

def get_combined_suggestions(predicted_traits):
    sentences = []
    for trait, level in predicted_traits.items():
        if trait in suggestions_dict:
            text = suggestions_dict[trait].get(level, "")
            if text:
                sentences.append(text)
    return " ".join(sentences) + "."

if __name__ == "__main__":
    models = load_models()
    img_path = input("Enter path of handwriting image to predict: ").strip()
    preds = predict_image(models, img_path)

    print("\n📝 Predicted Traits:")
    for t, v in preds.items():
        print(f"{t}: {v}")

    print("\n💡 Suggestions:")
    print(get_combined_suggestions(preds))

    print("\n📝 Note: Please meet a counselor or guide for further advice.")