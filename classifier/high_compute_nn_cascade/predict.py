"""
predict.py

Runs the full high_compute_nn_cascade pipeline on a single new video:
    1. Extract frames
    2. Embed with CLIP
    3. Stage 1 MLP: is it surgery?
    4. If yes -> Stage 2 MLP: is it cataract surgery?

Final label is one of: "not_surgery", "surgery_other" (any other surgery
type), "cataract".

Usage:
    python predict.py --video path/to/clip.mp4 --model_dir models/
"""

import argparse
import json
import os
import tempfile

import numpy as np
import torch
from joblib import load

from embed_frames import embed_video_frames, load_clip
from extract_frames import extract_frames_from_video
from train_nn_cascade import StageMLP


def _load_stage(model_dir: str, stage_file: str, scaler_file: str, device):
    checkpoint = torch.load(os.path.join(model_dir, stage_file), map_location=device)
    model = StageMLP(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    scaler = load(os.path.join(model_dir, scaler_file))
    return model, scaler


@torch.no_grad()
def _predict_proba(model, scaler, X: np.ndarray, device):
    X_scaled = scaler.transform(X)
    logits = model(torch.tensor(X_scaled, dtype=torch.float32).to(device))
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def classify_video(video_path: str, model_dir: str, n_frames: int = 8, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    stage1_model, stage1_scaler = _load_stage(model_dir, "stage1_is_surgery.pt", "stage1_scaler.joblib", device)
    stage2_model, stage2_scaler = _load_stage(model_dir, "stage2_is_cataract.pt", "stage2_scaler.joblib", device)

    clip_model, preprocess = load_clip(device)

    with tempfile.TemporaryDirectory() as tmp_dir:
        extract_frames_from_video(video_path, tmp_dir, n_frames)
        frame_paths = [
            os.path.join(tmp_dir, f) for f in sorted(os.listdir(tmp_dir))
        ]
        if not frame_paths:
            raise RuntimeError(f"No frames extracted from {video_path}")

        embedding = embed_video_frames(frame_paths, clip_model, preprocess, device)

    X = embedding.reshape(1, -1)

    stage1_probs = _predict_proba(stage1_model, stage1_scaler, X, device)
    stage1_pred = int(stage1_probs.argmax())

    result = {
        "video": video_path,
        "is_surgery": bool(stage1_pred),
        "is_surgery_confidence": float(stage1_probs.max()),
    }

    if stage1_pred == 1:
        stage2_probs = _predict_proba(stage2_model, stage2_scaler, X, device)
        stage2_pred = int(stage2_probs.argmax())
        result["is_cataract"] = bool(stage2_pred)
        result["is_cataract_confidence"] = float(stage2_probs.max())
        result["final_label"] = "cataract" if stage2_pred == 1 else "surgery_other"
    else:
        result["final_label"] = "not_surgery"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--n_frames", type=int, default=8)
    args = parser.parse_args()

    result = classify_video(args.video, args.model_dir, args.n_frames)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
