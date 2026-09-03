"""
train_trees_cascade.py

stacked_trees_cascade: trains the two-stage cascade using GRADIENT-BOOSTED
DECISION TREES for each stage (this is the folder's namesake — previously
this script trained plain logistic regression, which isn't a tree model at
all; see the NOTE at the bottom of this file for that history):

    Stage 1: is_surgery       (all videos, binary)
    Stage 2: is_cataract      (surgery videos only, binary — other surgery
                                types are the hard negatives here)

Each stage is an independent tree-ensemble classifier trained on the
pooled CLIP embeddings from embed_frames.py. Stages are trained separately
so you can debug/evaluate each one on its own before chaining them at
inference time.

Expects a labels CSV with columns: video_id, is_surgery, surgery_type
  - is_surgery: 0 or 1
  - surgery_type: "cataract", "other_surgery", or blank if is_surgery=0

Usage:
    python train_trees_cascade.py --embeddings embeddings.parquet --labels labels.csv --out_dir models/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data(embeddings_path: str, labels_path: str) -> pd.DataFrame:
    """
    Load and merge the embeddings and labels into a single DataFrame.

    - embeddings_path: parquet file with columns [video_id, embedding]
    - labels_path: CSV file with columns [video_id, is_surgery, surgery_type]

    Returns a merged DataFrame containing only videos that have BOTH
    an embedding and a label (inner join).
    """
    emb_df = pd.read_parquet(embeddings_path)
    labels_df = pd.read_csv(labels_path)

    # Inner join: keep only rows where video_id exists in both files.
    df = emb_df.merge(labels_df, on="video_id", how="inner")

    # Warn (don't fail) if some labeled videos got dropped because
    # they have no corresponding embedding — helps catch data pipeline bugs.
    missing = set(labels_df["video_id"]) - set(emb_df["video_id"])
    if missing:
        print(f"[warn] {len(missing)} labeled videos have no embedding, skipping: "
              f"{list(missing)[:5]}...")
    return df


def _sample_weights(y: np.ndarray) -> np.ndarray:
    """
    GradientBoostingClassifier has no built-in class_weight="balanced"
    option (unlike LogisticRegression/RandomForest), so we compute
    per-sample weights manually and pass them via sample_weight= at fit
    time to get the same class-balancing behavior.
    """
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    n = len(y)
    weight_per_class = {c: n / (len(classes) * cnt) for c, cnt in freq.items()}
    return np.array([weight_per_class[label] for label in y])


def safe_split(X, y, test_size=0.2, random_state=42):
    """
    Stratified train/val split that falls back to a plain (unstratified)
    split when a class has too few members to stratify — this matters
    for small smoke-test runs (e.g. orchestrator --test with 5 videos),
    where stratify=y would otherwise raise ValueError.
    """
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as e:
        print(f"[warn] stratified split failed ({e}); falling back to a plain split "
              f"(expected on tiny/test-mode datasets)")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_binary_stage(X: np.ndarray, y: np.ndarray, stage_name: str):
    """
    Generic trainer for a single binary classification stage.

    - X: feature matrix (n_samples, embedding_dim)
    - y: binary labels (0/1)
    - stage_name: human-readable name used in printed reports

    Splits data into train/val, fits a gradient-boosted tree ensemble,
    prints a validation report, and returns the fitted classifier.
    """
    # Hold out 20% of the data for validation, stratified so both
    # splits preserve the original class balance (important since
    # cataract vs. other-surgery is likely imbalanced).
    X_train, X_val, y_train, y_val = safe_split(X, y, test_size=0.2, random_state=42)

    sample_weight = _sample_weights(y_train)

    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    # Evaluate on the held-out validation set and print diagnostics
    # so each stage can be sanity-checked independently.
    preds = clf.predict(X_val)
    print(f"\n=== {stage_name} validation report ===")
    # labels=[0, 1] keeps this from crashing when a tiny (e.g. test-mode)
    # split doesn't happen to contain both classes.
    print(classification_report(y_val, preds, labels=[0, 1], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, preds, labels=[0, 1]))

    return clf


def main():
    # ---- CLI arguments ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)   # path to embeddings.parquet
    parser.add_argument("--labels", required=True)       # path to labels.csv
    parser.add_argument("--out_dir", required=True)       # where to save trained models
    args = parser.parse_args()

    # Make sure the output directory exists before we try to save models into it.
    os.makedirs(args.out_dir, exist_ok=True)

    # Load and merge embeddings + labels into one DataFrame.
    df = load_data(args.embeddings, args.labels)

    # Stack the embedding column (array of vectors) into a proper
    # 2D NumPy matrix of shape (n_videos, embedding_dim).
    X_all = np.stack(df["embedding"].values)

    # ---------- Stage 1: is_surgery ----------
    # Trained on ALL videos: surgery vs. non-surgery.
    y_stage1 = df["is_surgery"].values
    stage1_clf = train_binary_stage(X_all, y_stage1, "Stage 1 (is_surgery)")

    # Persist the trained Stage 1 model to disk for later inference.
    dump(stage1_clf, os.path.join(args.out_dir, "stage1_is_surgery.joblib"))

    # ---------- Stage 2: is_cataract (surgery videos only) ----------
    # Filter down to only the videos that Stage 1 would consider "surgery".
    surgery_df = df[df["is_surgery"] == 1].copy()

    # Sanity check: every surgery video must have a surgery_type label.
    # If not, the labels CSV is malformed — fail loudly rather than
    # silently mishandling NaNs during training.
    if surgery_df["surgery_type"].isna().any():
        raise ValueError("Found is_surgery=1 rows with missing surgery_type — fix labels.csv")

    # Build the feature matrix restricted to surgery videos only.
    X_surgery = np.stack(surgery_df["embedding"].values)

    # Binary label for Stage 2: 1 = cataract, 0 = any other surgery type
    # (these other-surgery videos serve as the "hard negatives" here,
    # since they look visually similar to cataract surgery).
    y_stage2 = (surgery_df["surgery_type"] == "cataract").astype(int).values

    # Print class balance so imbalance issues are visible before training.
    print(f"\nStage 2 class balance — cataract: {y_stage2.sum()}, "
          f"non-cataract surgeries: {(1 - y_stage2).sum()}")

    # Train and save Stage 2 the same way as Stage 1.
    stage2_clf = train_binary_stage(X_surgery, y_stage2, "Stage 2 (is_cataract)")
    dump(stage2_clf, os.path.join(args.out_dir, "stage2_is_cataract.joblib"))

    # ---------- Save metadata ----------
    # Records what each class index (0/1) means for each stage, so
    # inference code can interpret predict() outputs consistently
    # without hardcoding label meanings elsewhere.
    meta = {
        "model_type": "gradient_boosted_trees",
        "stage1_classes": ["not_surgery", "surgery"],
        "stage2_classes": ["not_cataract", "cataract"],
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModels saved to {args.out_dir}")


if __name__ == "__main__":
    main()

# NOTE ON HISTORY: this file used to be byte-for-byte identical to the
# train_cascade.py that now lives in high_compute_nn_cascade/ — both
# trained a plain sklearn LogisticRegression per stage, so neither
# script actually matched its own folder name ("stacked trees" wasn't
# using trees; "high compute NN" wasn't a neural net at all, and was
# also duplicated code rather than a distinct approach). This file was
# updated to use GradientBoostingClassifier so the folder's name is
# accurate. See classifier/README.md for the same fix applied to
# high_compute_nn_cascade/.
