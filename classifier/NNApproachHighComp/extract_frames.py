"""
extract_frames.py

Samples N frames uniformly spaced across each video and saves them as .jpg.

Why uniform sampling: cheap, and a strong baseline (used in Temporal Segment
Networks). If your error analysis later shows the model is confusing
appearance-similar-but-motion-different classes, switch to dense clip
sampling instead (see the note at the bottom of this file).

Usage:
    python extract_frames.py --video_dir raw_videos/ --out_dir frames/ --n_frames 8
"""

import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def sample_frame_indices(total_frames: int, n_frames: int) -> list[int]:
    """Evenly spaced frame indices across the video, avoiding the very first/last frame."""
    if total_frames <= n_frames:
        return list(range(total_frames))
    # evenly spaced, offset slightly inward
    step = total_frames / (n_frames + 1)
    return [int(step * (i + 1)) for i in range(n_frames)]


def extract_frames_from_video(video_path: str, out_dir: str, n_frames: int) -> int:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        print(f"  [warn] could not read frame count for {video_path}, skipping")
        return 0

    indices = sample_frame_indices(total_frames, n_frames)
    video_stem = Path(video_path).stem
    saved = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = os.path.join(out_dir, f"{video_stem}_f{idx:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Directory of input videos")
    parser.add_argument("--out_dir", required=True, help="Directory to write extracted frames")
    parser.add_argument("--n_frames", type=int, default=8, help="Frames to sample per video")
    parser.add_argument(
        "--extensions", nargs="+", default=[".mp4", ".mov", ".avi", ".mkv"]
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    video_paths = [
        os.path.join(args.video_dir, f)
        for f in os.listdir(args.video_dir)
        if Path(f).suffix.lower() in args.extensions
    ]

    print(f"Found {len(video_paths)} videos in {args.video_dir}")

    total_saved = 0
    for vp in tqdm(video_paths, desc="Extracting frames"):
        total_saved += extract_frames_from_video(vp, args.out_dir, args.n_frames)

    print(f"Done. Saved {total_saved} frames to {args.out_dir}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# NOTE on switching to dense/motion-aware sampling later:
# Instead of N independent frames, extract a contiguous clip of ~16-32 frames
# around a few anchor points in the video, and feed the clip (not individual
# frames) into a video-native model (VideoMAE, X-CLIP, SlowFast) rather than
# a per-frame image model. The extraction logic changes to: pick K anchor
# timestamps, then grab frames[anchor : anchor + clip_len] at each anchor.
# ---------------------------------------------------------------------------
