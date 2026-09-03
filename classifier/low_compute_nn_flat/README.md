# Low-Compute NN Approach — Flat MLP Classifier

A single lightweight neural net (256 -> 64 -> 3) trained directly on
pooled CLIP frame embeddings. This is the "low compute" NN approach:
one small model, one training run — no chained/cascaded stages.

## Pipeline

```
raw_videos/  --extract_frames.py-->  frames/  --embed_frames.py-->  embeddings.parquet
                                                                          |
                                                    labels.csv -----> train_flat_mlp.py --> models_mlp/
                                                                          |
                                                                       predict.py (embeddings -> labels)
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
classes are hard to tell apart from a handful of frames (e.g. if motion
matters more than a single snapshot).

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

## 4. Train the flat MLP

```bash
python train_flat_mlp.py --embeddings embeddings.parquet --labels labels.csv --out_dir models_mlp/
```

Collapses `is_surgery` + `surgery_type` into a single 3-class target
(`not_surgery`, `surgery_not_cataract`, `cataract`) and trains **one**
MLP with class-weighted loss, early stopping, and a held-out test set.
Prints a classification report + confusion matrix on the test split.

Unlike the cascade approaches, there's no stage-1/stage-2 split here —
the model predicts all 3 classes directly in a single forward pass,
which is cheaper to train and simpler to reason about, at the cost of
not being able to debug/tune each decision boundary independently.

## 5. Predict on new embeddings

```bash
python predict.py --embeddings new_embeddings.parquet --model_dir models_mlp/ --out_csv predictions.csv
```

Outputs a CSV with `video_id`, `prediction`, `confidence`, and
per-class probabilities (`prob_not_surgery`, `prob_surgery_not_cataract`,
`prob_cataract`).
