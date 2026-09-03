"""
predict.py (low_compute_nn_flat)

Loads the trained flat MLP + scaler (produced by train_flat_mlp.py) and
classifies new videos from their pre-computed CLIP embeddings.

Usage:
    python predict.py --embeddings new_embeddings.parquet \
        --model_dir models_mlp/ --out_csv predictions.csv
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from joblib import load
from train_flat_mlp import MLPClassifier  # reuse the class definition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_csv", default="predictions.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- load trained model + scaler ----
    checkpoint = torch.load(f"{args.model_dir}/mlp_classifier.pt", map_location=device)
    scaler = load(f"{args.model_dir}/scaler.joblib")
    class_names = checkpoint["class_names"]

    model = MLPClassifier(input_dim=checkpoint["input_dim"], num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # ---- load new embeddings to classify ----
    df = pd.read_parquet(args.embeddings)
    X = np.stack(df["embedding"].values)

    # IMPORTANT: use the scaler fitted during training (transform, not
    # fit_transform) — this must match training-time preprocessing exactly,
    # otherwise predictions will be silently wrong.
    X = scaler.transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # ---- run inference ----
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    # ---- attach predictions + confidence scores to the dataframe ----
    df["prediction"] = [class_names[p] for p in preds]
    df["confidence"] = probs.max(axis=1)

    # also save the full per-class probability breakdown — useful for
    # spot-checking uncertain predictions later (e.g. cataract vs.
    # surgery_not_cataract might be close for borderline videos)
    for i, cname in enumerate(class_names):
        df[f"prob_{cname}"] = probs[:, i]

    # ---- save results ----
    out_cols = ["video_id", "prediction", "confidence"] + [f"prob_{c}" for c in class_names]
    out_cols = [c for c in out_cols if c in df.columns]  # guard in case video_id is missing
    df[out_cols].to_csv(args.out_csv, index=False)

    print(f"\nSaved {len(df)} predictions to {args.out_csv}")
    print("\nPrediction distribution:")
    print(df["prediction"].value_counts())

    print("\nSample predictions:")
    print(df[out_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()