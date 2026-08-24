"""
predict.py

Runs the full cascade on a single new video:
    1. Extract frames
    2. Embed with CLIP
    3. Stage 1: is it a sport?
    4. If yes -> Stage 2: is it baseball?

Final label is one of: "not_sport", "sport_other" (includes cricket etc.),
"baseball".

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


def classify_video(video_path: str, model_dir: str, n_frames: int = 8, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    stage1_clf = load(os.path.join(model_dir, "stage1_is_sport.joblib"))
    stage2_clf = load(os.path.join(model_dir, "stage2_is_baseball.joblib"))

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

    stage1_pred = stage1_clf.predict(X)[0]
    stage1_prob = stage1_clf.predict_proba(X)[0]

    result = {
        "video": video_path,
        "is_sport": bool(stage1_pred),
        "is_sport_confidence": float(max(stage1_prob)),
    }

    if stage1_pred == 1:
        stage2_pred = stage2_clf.predict(X)[0]
        stage2_prob = stage2_clf.predict_proba(X)[0]
        result["is_baseball"] = bool(stage2_pred)
        result["is_baseball_confidence"] = float(max(stage2_prob))
        result["final_label"] = "baseball" if stage2_pred == 1 else "sport_other"
    else:
        result["final_label"] = "not_sport"

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
