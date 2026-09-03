# Stacked Trees Cascade Approach

A two-stage cascade for hierarchical video classification, where each
stage is a gradient-boosted decision tree ensemble trained on pooled
CLIP frame embeddings. This is the "trees" approach: no neural nets are
trained here — `scikit-learn`'s `GradientBoostingClassifier` does the
classification work at both stages.

## Pipeline

```
raw_videos/  --extract_frames.py-->  frames/  --embed_frames.py-->  embeddings.parquet
                                                                          |
                                                    labels.csv -----> train_trees_cascade.py --> models/
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
python train_trees_cascade.py --embeddings embeddings.parquet --labels labels.csv --out_dir models/
```

Trains stage 1 (`is_surgery`) on all videos and stage 2 (`is_cataract`)
on surgery-only videos, each as a `GradientBoostingClassifier`. Prints a
classification report + confusion matrix for each stage — check these
before trusting the model.

Why trees here: gradient-boosted trees can pick up non-linear decision
boundaries in the CLIP embedding space that a linear model (see
`low_compute_nn_flat/` or a logistic-regression baseline) would miss,
usually at a moderate compute cost — more expensive than logistic
regression, cheaper than training a neural net from these embeddings.

## 5. Predict on a new video

```bash
python predict.py --video new_clip.mp4 --model_dir models/
```

Outputs JSON with the final label (`not_surgery` / `surgery_other` /
`cataract`) and per-stage confidence.

## Changelog

This folder previously contained a script (`train_cascade.py`) that was
a byte-for-byte duplicate of the one in `high_compute_nn_cascade/` —
both trained plain `LogisticRegression` models, so neither one actually
used trees, and `predict.py` referenced leftover class names
(`is_sport` / `is_baseball`) from an unrelated template. Both issues are
fixed here: this folder now genuinely trains tree ensembles, and
`predict.py` matches the surgery/cataract labels this repo actually
uses.
