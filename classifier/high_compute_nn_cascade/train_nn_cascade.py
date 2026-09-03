"""
train_nn_cascade.py

high_compute_nn_cascade: trains the two-stage cascade using an actual
neural network at EACH stage (this is the folder's namesake — previously
this script trained plain sklearn LogisticRegression, which is neither a
neural net nor "high compute"; see the NOTE at the bottom of this file
for that history):

    Stage 1: is_surgery       (all videos, binary) -> MLP #1
    Stage 2: is_cataract      (surgery videos only, binary) -> MLP #2

Each stage is an independently-trained MLP (512 -> 256 -> 128 -> 2) on
the pooled CLIP embeddings from embed_frames.py. This is intentionally
a bigger network with more layers than the single flat model in
low_compute_nn_flat/ (256 -> 64 -> 3), and trains TWO of them instead of
one — that's the "high compute" side of the comparison: more parameters,
two training runs instead of one, longer wall-clock time. Whether the
extra compute is worth it over the flat MLP or the tree cascade is
exactly the kind of thing this repo's 3-way comparison is meant to show.

Expects a labels CSV with columns: video_id, is_surgery, surgery_type
  - is_surgery: 0 or 1
  - surgery_type: "cataract", "other_surgery", or blank if is_surgery=0

Usage:
    python train_nn_cascade.py --embeddings embeddings.parquet --labels labels.csv --out_dir models/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump


class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StageMLP(nn.Module):
    """embedding_dim -> 512 -> 256 -> 128 -> 2 classes.

    Deliberately larger than low_compute_nn_flat's MLPClassifier
    (256 -> 64) so this folder's name ("high compute") is backed up by
    an actually heavier model, not just a different label on the same
    architecture.
    """
    def __init__(self, input_dim: int, hidden=(512, 256, 128), dropout: float = 0.3):
        super().__init__()
        h1, h2, h3 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h3, 2),
        )

    def forward(self, x):
        return self.net(x)


def safe_split(X, y, test_size, random_state=42):
    """
    Stratified split that falls back to a plain (unstratified) split when
    a class has too few members to stratify — matters for small smoke
    tests (e.g. orchestrator --test with 5 videos), where stratify=y
    would otherwise raise ValueError.
    """
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError as e:
        print(f"[warn] stratified split failed ({e}); falling back to a plain split "
              f"(expected on tiny/test-mode datasets)")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_data(embeddings_path: str, labels_path: str) -> pd.DataFrame:
    emb_df = pd.read_parquet(embeddings_path)
    labels_df = pd.read_csv(labels_path)
    df = emb_df.merge(labels_df, on="video_id", how="inner")
    missing = set(labels_df["video_id"]) - set(emb_df["video_id"])
    if missing:
        print(f"[warn] {len(missing)} labeled videos have no embedding, skipping: "
              f"{list(missing)[:5]}...")
    return df


def train_stage_mlp(X: np.ndarray, y: np.ndarray, stage_name: str, device,
                     epochs: int = 60, batch_size: int = 32, lr: float = 1e-3,
                     patience: int = 8):
    """
    Trains one binary-classification MLP stage with a train/val/test
    split, class-weighted loss, LR scheduling, and early stopping.
    Prints a held-out test report and returns (model, scaler, input_dim).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = safe_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = safe_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(EmbeddingDataset(X_test, y_test), batch_size=batch_size)

    class_counts = np.bincount(y_train, minlength=2)
    class_weights = torch.tensor(len(y_train) / (2 * np.maximum(class_counts, 1)), dtype=torch.float32)

    model = StageMLP(input_dim=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                loss = criterion(model(X_batch), y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{stage_name}] Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    print(f"\n=== {stage_name} test report ===")
    # labels=[0, 1] keeps this from crashing when a tiny (e.g. test-mode)
    # split doesn't happen to contain both classes.
    print(classification_report(all_labels, all_preds, labels=[0, 1], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return model, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_data(args.embeddings, args.labels)
    X_all = np.stack(df["embedding"].values)

    # ---------- Stage 1: is_surgery ----------
    y_stage1 = df["is_surgery"].values
    stage1_model, stage1_scaler = train_stage_mlp(
        X_all, y_stage1, "Stage 1 (is_surgery)", device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    torch.save(
        {"model_state_dict": stage1_model.state_dict(), "input_dim": X_all.shape[1]},
        os.path.join(args.out_dir, "stage1_is_surgery.pt"),
    )
    dump(stage1_scaler, os.path.join(args.out_dir, "stage1_scaler.joblib"))

    # ---------- Stage 2: is_cataract (surgery videos only) ----------
    surgery_df = df[df["is_surgery"] == 1].copy()
    if surgery_df["surgery_type"].isna().any():
        raise ValueError("Found is_surgery=1 rows with missing surgery_type — fix labels.csv")

    X_surgery = np.stack(surgery_df["embedding"].values)
    y_stage2 = (surgery_df["surgery_type"] == "cataract").astype(int).values
    print(f"\nStage 2 class balance — cataract: {y_stage2.sum()}, "
          f"non-cataract surgeries: {(1 - y_stage2).sum()}")

    stage2_model, stage2_scaler = train_stage_mlp(
        X_surgery, y_stage2, "Stage 2 (is_cataract)", device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    torch.save(
        {"model_state_dict": stage2_model.state_dict(), "input_dim": X_surgery.shape[1]},
        os.path.join(args.out_dir, "stage2_is_cataract.pt"),
    )
    dump(stage2_scaler, os.path.join(args.out_dir, "stage2_scaler.joblib"))

    meta = {
        "model_type": "mlp_cascade",
        "stage1_classes": ["not_surgery", "surgery"],
        "stage2_classes": ["not_cataract", "cataract"],
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModels saved to {args.out_dir}")


if __name__ == "__main__":
    main()

# NOTE ON HISTORY: this file used to be byte-for-byte identical to the
# train_cascade.py that now lives in stacked_trees_cascade/ — both
# trained a plain sklearn LogisticRegression per stage, so neither
# script matched its own folder name ("high compute NN" wasn't a neural
# net at all, and "stacked trees" wasn't using trees). This file now
# trains a genuine two-stage MLP cascade with more parameters and a
# real training loop, so the folder's name is accurate. See
# classifier/README.md for the same fix applied to
# stacked_trees_cascade/.
