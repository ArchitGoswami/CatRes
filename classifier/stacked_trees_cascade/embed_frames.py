"""
embed_frames.py

Runs a pretrained CLIP image encoder over extracted frames and average-pools
per video to get one embedding vector per video. This embedding is the input
to the cascade classifiers trained in train_cascade.py.

Why CLIP: strong general-purpose visual representations, zero training
needed for the backbone, and fast to iterate with. Swap in a video-native
encoder later if frame-level appearance turns out not to be enough.

Usage:
    python embed_frames.py --frame_dir frames/ --out_file embeddings.parquet
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Matches "<video_stem>_f<frame_idx>.jpg" produced by extract_frames.py
FRAME_PATTERN = re.compile(r"^(.*)_f\d+\.jpg$")


def group_frames_by_video(frame_dir: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for f in sorted(Path(frame_dir).glob("*.jpg")):
        match = FRAME_PATTERN.match(f.name)
        if match:
            video_id = match.group(1)
            groups[video_id].append(str(f))
    return groups


def load_clip(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval().to(device)
    return model, preprocess


@torch.no_grad()
def embed_video_frames(frame_paths: list[str], model, preprocess, device: str) -> np.ndarray:
    imgs = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        imgs.append(preprocess(img))
    batch = torch.stack(imgs).to(device)
    features = model.encode_image(batch)
    features = features / features.norm(dim=-1, keepdim=True)  # normalize
    pooled = features.mean(dim=0)  # average pool across sampled frames
    pooled = pooled / pooled.norm()  # re-normalize after pooling
    return pooled.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", required=True)
    parser.add_argument("--out_file", required=True, help="Output .parquet path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    model, preprocess = load_clip(args.device)

    video_groups = group_frames_by_video(args.frame_dir)
    print(f"Found {len(video_groups)} videos worth of frames")

    rows = []
    for video_id, frame_paths in tqdm(video_groups.items(), desc="Embedding videos"):
        emb = embed_video_frames(frame_paths, model, preprocess, args.device)
        rows.append({"video_id": video_id, "embedding": emb.tolist()})

    df = pd.DataFrame(rows)
    df.to_parquet(args.out_file)
    print(f"Saved {len(df)} embeddings to {args.out_file}")


if __name__ == "__main__":
    main()
