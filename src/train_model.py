import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "features.csv")
TRAIT_LABELS_CSV = os.path.join(BASE_DIR, "trait_labels.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVAL_REPORTS_DIR = os.path.join(BASE_DIR, "eval_reports")

trait_features_map = {
    "Academic":        ["neatness", "spacing", "baseline_shift", "letter_size"],
    "Confidence":      ["slant", "letter_size", "pressure", "connection_type"],
    "Creativity":      ["loop_size", "slant", "stroke_variation", "baseline_shift"],
    "Drawing":         ["loop_size", "stroke_variation", "pressure", "letter_size"],
    "Public_Speaking": ["slant", "spacing", "connection_type", "stroke_speed"],
    "Sports":          ["pressure", "stroke_speed", "letter_size"]
}

def train_and_evaluate(features_df: pd.DataFrame, labels_df: pd.DataFrame, save_models=True):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVAL_REPORTS_DIR, exist_ok=True)

    for trait, cols in trait_features_map.items():
        if trait not in labels_df.columns:
            print(f"⚠️ No labels for trait '{trait}', skipping.")
            continue
        missing = [c for c in cols if c not in features_df.columns]
        if missing:
            print(f"⚠️ {trait}: missing features {missing}, skipping.")
            continue

        X = features_df[cols].copy()
        y = labels_df[trait].astype(str)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipe = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=42,
                class_weight="balanced_subsample", n_jobs=-1
            )
        )
        pipe.fit(X_tr, y_tr)
        y_pr = pipe.predict(X_te)

        acc = accuracy_score(y_te, y_pr)
        cm = confusion_matrix(y_te, y_pr, labels=["Low","Moderate","High"])
        report = classification_report(y_te, y_pr, labels=["Low","Moderate","High"])

        print(f"\n===== {trait} =====")
        print(f"Accuracy: {acc:.3f}")
        print("Confusion Matrix (rows=actual, cols=pred):\n", cm)
        print("Classification Report:\n", report)

        plt.figure(figsize=(4.2, 3.6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Low","Moderate","High"],
                    yticklabels=["Low","Moderate","High"])
        plt.title(f"{trait} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = os.path.join(EVAL_REPORTS_DIR, f"{trait}_cm.png")
        plt.savefig(cm_path, dpi=140)
        plt.close('all')

        f1 = f1_score(y_te, y_pr, labels=["Low","Moderate","High"], average=None)
        precision = precision_score(y_te, y_pr, labels=["Low","Moderate","High"], average=None)
        recall = recall_score(y_te, y_pr, labels=["Low","Moderate","High"], average=None)

        x = np.arange(len(["Low","Moderate","High"]))
        width = 0.2

        plt.figure(figsize=(10, 5))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.85, edgecolor='black', linewidth=0.5)
        plt.bar(x, recall, width, label='Recall', alpha=0.85, edgecolor='black', linewidth=0.5)
        plt.bar(x + width, f1, width, label='F1-score', alpha=0.85, edgecolor='black', linewidth=0.5)

        plt.xticks(x, ["Low", "Moderate", "High"], fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.15)
        plt.title(f"{trait} Model Performance Metrics", fontsize=14, fontweight='bold', pad=15)
        plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        metric_path = os.path.join(EVAL_REPORTS_DIR, f"{trait}_metrics.png")
        plt.savefig(metric_path, dpi=150, bbox_inches='tight')
        plt.close('all')

        with open(os.path.join(EVAL_REPORTS_DIR, f"{trait}_report.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(report)

        if save_models:
            model_path = os.path.join(MODELS_DIR, f"{trait.lower()}_model.pkl")
            joblib.dump(pipe, model_path)

        print(f"💾 Saved model → {model_path}")
        print(f"🖼  Saved CM heatmap → {cm_path}")
        print(f"📊 Saved F1-score chart → {metric_path}")
        print(f"📝 Saved report → eval_reports/{trait}_report.txt")

if __name__ == "__main__":
    if not os.path.exists(FEATURES_CSV):
        print("❌ features.csv not found. Run feature_extractor.py first.")
        raise SystemExit(1)
    if not os.path.exists(TRAIT_LABELS_CSV):
        print("❌ trait_labels.csv not found. Run clustering.py first.")
        raise SystemExit(1)

    feats = pd.read_csv(FEATURES_CSV, index_col="image_name")
    labels = pd.read_csv(TRAIT_LABELS_CSV, index_col="image_name")
    train_and_evaluate(feats, labels, save_models=True)