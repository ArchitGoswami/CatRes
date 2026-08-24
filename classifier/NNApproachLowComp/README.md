# Video Cascade Classifier

A two-stage cascade for hierarchical video classification using pooled CLIP frame embeddings as a fast, strong baseline.

## Pipeline

```
raw_videos/  --extract_frames.py-->  frames/  --embed_frames.py-->  embeddings.parquet
                                                                          |
                                                    labels.csv -----> train_cascade.py --> models/
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
classes are hard to tell apart from a handful of frames (e.g. if motion
matters more than a single snapshot).

Why extracting frames?

Sparse frame sampling: Wang et al. (2016), "Temporal Segment Networks: Towards Good Practices for Deep Action Recognition," ECCV — this is the paper that established sparse uniform sampling as standard practice. Key finding: consecutive frames are highly redundant, so dense temporal sampling — which usually results in highly similar sampled frames — is unnecessary; a sparse sampling scheme with samples distributed uniformly along the temporal dimension works about as well

## 2. Embed frames

```bash
python embed_frames.py --frame_dir frames/ --out_file embeddings.parquet
```

Runs each frame through CLIP (ViT-B/32) and average-pools the per-frame
embeddings into a single vector per video.

## 3. Label your data

Fill out a CSV like `labels_template.csv`

## 4. Train the cascade

```bash
python train_cascade.py --embeddings embeddings.parquet --labels labels.csv --out_dir models/
```

Trains stage 1 (`is_surgery`) on all videos and stage 2 (`is_cataract`) on
surgery-only videos. Prints a classification report + confusion matrix for
each stage — check these before trusting the model.

## 5. Predict on a new video

```bash
python predict.py --video new_clip.mp4 --model_dir models/
```

Outputs JSON with the final label (`not_surgery` / `surgery_other` / `cataract`)
and per-stage confidence.