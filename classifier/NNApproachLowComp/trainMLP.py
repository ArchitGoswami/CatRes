"""
train_mlp_option_a.py

Flat 3-class MLP classifier trained on frozen CLIP embeddings
(embeddings.parquet, produced by embed_frames.py — same input as
train_cascade.py, but here we train ONE model with 3 outputs instead
of two chained binary models).

Classes:
    0 = not_surgery
    1 = surgery_not_cataract
    2 = cataract

Usage:
    python train_mlp_option_a.py --embeddings embeddings.parquet \
        --labels labels.csv --out_dir models_mlp/
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


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse is_surgery + surgery_type into a single 3-class target."""
    def label_row(row):
        if row["is_surgery"] == 0:
            return 0
        elif row["surgery_type"] == "cataract":
            return 2
        else:
            return 1

    df = df.copy()
    df["label"] = df.apply(label_row, axis=1)
    return df


class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    """embedding_dim -> 256 -> 64 -> 3 classes."""
    def __init__(self, input_dim: int, num_classes: int = 3,
                 hidden1: int = 256, hidden2: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, val_loader, class_weights, device,
                 epochs=50, lr=1e-3, patience=7):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:02d} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    emb_df = pd.read_parquet(args.embeddings)
    labels_df = pd.read_csv(args.labels)
    df = emb_df.merge(labels_df, on="video_id", how="inner")
    df = build_labels(df)

    class_names = ["not_surgery", "surgery_not_cataract", "cataract"]
    print("Class distribution:\n", df["label"].value_counts().sort_index())

    X = np.stack(df["embedding"].values)
    y = df["label"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train_loader = DataLoader(EmbeddingDataset(X_train, y_train),
                               batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(EmbeddingDataset(X_test, y_test), batch_size=args.batch_size)

    class_counts = np.bincount(y_train, minlength=3)
    class_weights = torch.tensor(len(y_train) / (3 * class_counts), dtype=torch.float32)
    print("Class weights:", class_weights)

    model = MLPClassifier(input_dim=X.shape[1], num_classes=3).to(device)
    model = train_model(model, train_loader, val_loader, class_weights, device,
                         epochs=args.epochs, lr=args.lr)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    print("\n=== Final test set report ===")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": X.shape[1],
        "class_names": class_names,
    }, os.path.join(args.out_dir, "mlp_classifier.pt"))

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"class_names": class_names}, f, indent=2)

    print(f"\nModel saved to {args.out_dir}")


if __name__ == "__main__":
    main()