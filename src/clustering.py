import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "features.csv")
TRAIT_LABELS_CSV = os.path.join(BASE_DIR, "trait_labels.csv")
CLUSTER_MAPPINGS_DIR = os.path.join(BASE_DIR, "cluster_mappings")

trait_features_map = {
    "Academic":        ["neatness", "spacing", "baseline_shift", "letter_size"],
    "Confidence":      ["slant", "letter_size", "pressure", "connection_type"],
    "Creativity":      ["loop_size", "slant", "stroke_variation", "baseline_shift"],
    "Drawing":         ["loop_size", "stroke_variation", "pressure", "letter_size"],
    "Public_Speaking": ["slant", "spacing", "connection_type", "stroke_speed"],
    "Sports":          ["pressure", "stroke_speed", "letter_size"]
}

def cluster_traits(features_df: pd.DataFrame, save_csv=True):
    if features_df.empty:
        raise ValueError("features_df is empty. Run feature_extractor first.")

    os.makedirs(CLUSTER_MAPPINGS_DIR, exist_ok=True)
    trait_labels = pd.DataFrame(index=features_df.index)

    for trait, cols in trait_features_map.items():
        missing = [c for c in cols if c not in features_df.columns]
        if missing:
            print(f"⚠️ {trait}: missing columns {missing} → skipping")
            continue

        X = features_df[cols].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=3, random_state=42, n_init=20)
        cluster_ids = km.fit_predict(X_scaled)

        cluster_means = np.array([X.iloc[cluster_ids == k].mean().mean() for k in range(3)])
        order = np.argsort(cluster_means)

        consistent_mapping = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
        levels = pd.Series(cluster_ids, index=X.index).map(consistent_mapping)
        string_labels = levels.map({0: "Low", 1: "Moderate", 2: "High"})
        trait_labels[trait] = string_labels

        mapping_path = os.path.join(CLUSTER_MAPPINGS_DIR, f"{trait}.json")
        with open(mapping_path, "w") as f:
            json.dump(
                {
                    "cluster_id_to_level_index": consistent_mapping,
                    "level_index_to_label": {"0": "Low", "1": "Moderate", "2": "High"},
                    "cluster_means_low_to_high": cluster_means[order].tolist(),
                },
                f,
                indent=2,
            )

        print(f"✅ {trait}: label distribution →\n{string_labels.value_counts()}\n")

    if save_csv:
        trait_labels.to_csv(TRAIT_LABELS_CSV, index_label="image_name")
        print(f"✅ Saved trait_labels.csv ({trait_labels.shape[0]} rows)")

    print("\n📊 Final Trait Label Summary:")
    print(trait_labels.apply(pd.Series.value_counts).fillna(0).astype(int))

    return trait_labels

if __name__ == "__main__":
    if not os.path.exists(FEATURES_CSV):
        print("❌ features.csv not found. Run feature_extractor.py first.")
        raise SystemExit(1)

    feats = pd.read_csv(FEATURES_CSV, index_col="image_name")
    cluster_traits(feats, save_csv=True)