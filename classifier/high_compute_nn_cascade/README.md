# High-Compute NN Cascade Approach

A two-stage cascade for hierarchical video classification where EACH
stage is its own neural net (a 4-layer MLP: 512 -> 256 -> 128 -> 2)
trained on pooled CLIP frame embeddings. This is the heaviest of the
three classifier approaches in this repo — two separate networks, each
with early stopping, LR scheduling, and more parameters than the single
flat model in `low_compute_nn_flat/`.

## Pipeline

```
raw_videos/  --extract_frames.py-->  frames/  --embed_frames.py-->  embeddings.parquet
                                                                          |
                                                    labels.csv -----> train_nn_cascade.py --> models/
                                                                          |
                                                                       predict.py (new video -> label)
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Extract frames

```bash
python extract_frames.py --video_dir raw_videos/ --out_dir frames/ --n_frames 8
```

Samples 8 evenly-spaced frames per video. Increase `n_frames` if your
classes are hard to tell apart from a handful of frames.

## 2. Embed frames

```bash
python embed_frames.py --frame_dir frames/ --out_file embeddings.parquet
```

Runs each frame through CLIP (ViT-B/32) and average-pools the per-frame
embeddings into a single vector per video.

## 3. Label your data

Fill out a CSV with columns: `video_id, is_surgery, surgery_type`
(`surgery_type` is `cataract`, `other_surgery`, or blank when
`is_surgery=0`).

## 4. Train the cascade

```bash
python train_nn_cascade.py --embeddings embeddings.parquet --labels labels.csv --out_dir models/
```

Trains stage 1 (`is_surgery`, all videos) and stage 2 (`is_cataract`,
surgery-only videos) as two independent MLPs. Each stage does its own
train/val/test split, class-weighted `CrossEntropyLoss`, an LR
scheduler, and early stopping, then prints a held-out test report.

This is the most expensive of the three approaches to train (two neural
nets, GPU-friendly but slower than the tree cascade or the flat MLP) —
use it when the flat/tree baselines aren't separating `surgery_other`
from `cataract` well enough and you want more model capacity per
decision.

## 5. Predict on a new video

```bash
python predict.py --video new_clip.mp4 --model_dir models/
```

Outputs JSON with the final label (`not_surgery` / `surgery_other` /
`cataract`) and per-stage confidence.

## Changelog

This folder previously contained a script (`train_cascade.py`) that was
a byte-for-byte duplicate of the one in `stacked_trees_cascade/` — both
trained plain `LogisticRegression` models, so this folder wasn't
actually training a neural net at all, and `predict.py` referenced
leftover class names (`is_sport` / `is_baseball`) from an unrelated
template. Both issues are fixed here: this folder now trains a genuine
two-stage MLP cascade, and `predict.py` matches the surgery/cataract
labels this repo actually uses.
