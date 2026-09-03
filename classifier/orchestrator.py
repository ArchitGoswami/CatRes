"""
orchestrator.py

Runs the classifier pipeline end-to-end across all three approaches
(high_compute_nn_cascade, low_compute_nn_flat, stacked_trees_cascade):

    raw_videos/ + labels.csv
        --> extract_frames.py   (shared step, run once)
        --> embed_frames.py     (shared step, run once)
        --> train_*.py          (run once per approach, in parallel model dirs)
        --> report.md / report.json

Every run (test or full) gets its own timestamped directory under
--out_dir with frames/, embeddings.parquet, models/<approach>/, logs/,
and a report.

TEST MODE (--test)
-------------------
Before spending hours training on your full dataset, run with --test to
sanity-check the whole pipeline end-to-end on a small handful of videos
(5 by default, override with --test_n). This exercises every input/output
file path — frame extraction, embeddings, label joins, model artifacts —
without waiting on the full dataset. Look at the generated report before
re-running without --test on everything.

USAGE
-----
Full run, from raw video:
    python orchestrator.py --raw_videos raw_videos/ --labels labels.csv --out_dir runs/

Test run on 5 videos:
    python orchestrator.py --raw_videos raw_videos/ --labels labels.csv --out_dir runs/ --test

Skip extraction/embedding if you already have embeddings.parquet:
    python orchestrator.py --embeddings embeddings.parquet --labels labels.csv --out_dir runs/ --skip_extract --skip_embed

Only run specific approaches:
    python orchestrator.py --raw_videos raw_videos/ --labels labels.csv --approaches low_compute_nn_flat stacked_trees_cascade
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

APPROACHES = {
    "high_compute_nn_cascade": {
        "train_script": "train_nn_cascade.py",
        "description": "Two-stage cascade, each stage an independent MLP (heaviest to train).",
    },
    "low_compute_nn_flat": {
        "train_script": "train_flat_mlp.py",
        "description": "Single flat 3-class MLP, no cascade (cheapest neural approach).",
    },
    "stacked_trees_cascade": {
        "train_script": "train_trees_cascade.py",
        "description": "Two-stage cascade, each stage a gradient-boosted tree ensemble.",
    },
}

# extract_frames.py / embed_frames.py are duplicated identically across all
# three approach folders (a leftover of how this repo was scaffolded — see
# classifier/README.md). We only need to run them once per orchestrator run,
# so we call the copies that live in this folder by convention.
SHARED_SCRIPTS_SOURCE = "low_compute_nn_flat"


def run_cmd(cmd: list[str], log_path: Path, cwd: Path) -> tuple[int, float]:
    """Run a subprocess, streaming output to a log file. Returns (returncode, elapsed_seconds)."""
    start = time.time()
    with open(log_path, "w") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        log_file.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    return proc.returncode, elapsed


def make_test_subset(raw_videos: Path, labels_csv: Path, test_dir: Path, test_n: int,
                      extensions: list[str]) -> Path:
    """
    Build a small labels.csv + video folder covering as many distinct
    (is_surgery, surgery_type) combinations as are present in the full
    labels file, up to test_n videos total. This maximizes the chance
    that a 5-video test run still exercises both cascade stages instead
    of, say, sampling 5 videos that are all is_surgery=0.
    """
    labels_df = pd.read_csv(labels_csv)
    labels_df["_group"] = labels_df["surgery_type"].fillna("NONE") + "|" + labels_df["is_surgery"].astype(str)

    picked = []
    for _, group_df in labels_df.groupby("_group"):
        if len(picked) >= test_n:
            break
        picked.append(group_df.iloc[0])
    # fill remaining slots with whatever's left, in original order
    remaining = labels_df[~labels_df["video_id"].isin([r["video_id"] for r in picked])]
    for _, row in remaining.iterrows():
        if len(picked) >= test_n:
            break
        picked.append(row)

    subset_df = pd.DataFrame(picked).drop(columns=["_group"])
    test_dir.mkdir(parents=True, exist_ok=True)
    subset_labels_path = test_dir / "labels_test.csv"
    subset_df.to_csv(subset_labels_path, index=False)

    if raw_videos is not None:
        video_dir = test_dir / "raw_videos_sample"
        video_dir.mkdir(exist_ok=True)
        available = {p.stem: p for p in Path(raw_videos).iterdir() if p.suffix.lower() in extensions}
        missing = []
        for vid in subset_df["video_id"]:
            src = available.get(vid)
            if src is None:
                missing.append(vid)
                continue
            shutil.copy2(src, video_dir / src.name)
        if missing:
            print(f"[warn] {len(missing)} sampled video_ids have no matching file in "
                  f"{raw_videos}: {missing}")
        return subset_labels_path, video_dir

    return subset_labels_path, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_videos", help="Directory of raw video files (skip if using --embeddings)")
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument("--embeddings", help="Path to a precomputed embeddings.parquet (implies --skip_extract --skip_embed)")
    parser.add_argument("--out_dir", default="runs", help="Root directory for run outputs")
    parser.add_argument("--n_frames", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for NN-based trainers (has no effect on the trees approach)")
    parser.add_argument("--approaches", nargs="+", choices=list(APPROACHES.keys()), default=list(APPROACHES.keys()))
    parser.add_argument("--test", action="store_true", help="Sanity-check the full pipeline on a handful of videos before running on everything")
    parser.add_argument("--test_n", type=int, default=5, help="Number of videos to use in --test mode")
    parser.add_argument("--skip_extract", action="store_true", help="Skip frame extraction (frames/ already exists, or using --embeddings)")
    parser.add_argument("--skip_embed", action="store_true", help="Skip embedding (embeddings.parquet already exists, or using --embeddings)")
    parser.add_argument("--extensions", nargs="+", default=[".mp4", ".mov", ".avi", ".mkv"])
    args = parser.parse_args()

    if args.embeddings:
        args.skip_extract = True
        args.skip_embed = True

    if not args.embeddings and not args.raw_videos:
        parser.error("Provide either --raw_videos (to extract+embed) or --embeddings (precomputed)")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{'test_' if args.test else ''}run_{timestamp}"
    run_dir = Path(args.out_dir).resolve() / run_name
    logs_dir = run_dir / "logs"
    models_dir = run_dir / "models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_name": run_name,
        "started_at": timestamp,
        "test_mode": args.test,
        "config": vars(args).copy(),
        "steps": {},
        "approaches": {},
    }
    report["config"].pop("labels", None)  # keep report tidy; full paths logged separately
    print(f"=== Run: {run_name} (test_mode={args.test}) ===")
    print(f"Output dir: {run_dir}")

    labels_path = Path(args.labels)
    raw_videos_path = Path(args.raw_videos) if args.raw_videos else None

    # ---------- Optionally shrink to a small test subset first ----------
    if args.test:
        print(f"\n[test mode] Sampling up to {args.test_n} videos to validate the pipeline end-to-end...")
        subset_labels_path, subset_video_dir = make_test_subset(
            raw_videos_path, labels_path, run_dir / "test_subset", args.test_n, args.extensions
        )
        labels_path = subset_labels_path
        if subset_video_dir is not None:
            raw_videos_path = subset_video_dir
        n_sampled = len(pd.read_csv(labels_path))
        report["steps"]["test_subset"] = {"requested": args.test_n, "actual": n_sampled, "labels_path": str(labels_path)}
        print(f"[test mode] Using {n_sampled} videos for this run.")

    # ---------- Step 1: extract frames ----------
    frames_dir = run_dir / "frames"
    if not args.skip_extract:
        print("\n[1/3] Extracting frames...")
        script = REPO_ROOT / SHARED_SCRIPTS_SOURCE / "extract_frames.py"
        cmd = [
            sys.executable, str(script),
            "--video_dir", str(raw_videos_path),
            "--out_dir", str(frames_dir),
            "--n_frames", str(args.n_frames),
        ]
        rc, elapsed = run_cmd(cmd, logs_dir / "extract_frames.log", cwd=REPO_ROOT / SHARED_SCRIPTS_SOURCE)
        report["steps"]["extract_frames"] = {"returncode": rc, "elapsed_sec": round(elapsed, 2), "log": str(logs_dir / "extract_frames.log")}
        if rc != 0:
            _finish(report, run_dir, success=False, reason="extract_frames failed")
            sys.exit(1)
        print(f"  done in {elapsed:.1f}s")
    else:
        report["steps"]["extract_frames"] = {"skipped": True}

    # ---------- Step 2: embed frames ----------
    embeddings_path = Path(args.embeddings).resolve() if args.embeddings else (run_dir / "embeddings.parquet")
    if not args.skip_embed:
        print("\n[2/3] Embedding frames with CLIP...")
        script = REPO_ROOT / SHARED_SCRIPTS_SOURCE / "embed_frames.py"
        cmd = [
            sys.executable, str(script),
            "--frame_dir", str(frames_dir),
            "--out_file", str(embeddings_path),
        ]
        rc, elapsed = run_cmd(cmd, logs_dir / "embed_frames.log", cwd=REPO_ROOT / SHARED_SCRIPTS_SOURCE)
        report["steps"]["embed_frames"] = {"returncode": rc, "elapsed_sec": round(elapsed, 2), "log": str(logs_dir / "embed_frames.log")}
        if rc != 0:
            _finish(report, run_dir, success=False, reason="embed_frames failed")
            sys.exit(1)
        print(f"  done in {elapsed:.1f}s")
    else:
        report["steps"]["embed_frames"] = {"skipped": True, "embeddings_path": str(embeddings_path)}

    # ---------- Step 3: train each approach ----------
    print(f"\n[3/3] Training {len(args.approaches)} approach(es): {', '.join(args.approaches)}")
    any_failed = False
    for approach in args.approaches:
        info = APPROACHES[approach]
        approach_dir = REPO_ROOT / approach
        out_dir = models_dir / approach
        cmd = [
            sys.executable, info["train_script"],
            "--embeddings", str(embeddings_path),
            "--labels", str(labels_path),
            "--out_dir", str(out_dir),
        ]
        if args.epochs is not None and approach != "stacked_trees_cascade":
            cmd += ["--epochs", str(args.epochs)]

        print(f"  -> {approach} ({info['description']})")
        rc, elapsed = run_cmd(cmd, logs_dir / f"{approach}.log", cwd=approach_dir)
        status = "ok" if rc == 0 else "FAILED"
        print(f"     {status} in {elapsed:.1f}s (log: logs/{approach}.log)")
        report["approaches"][approach] = {
            "returncode": rc,
            "elapsed_sec": round(elapsed, 2),
            "log": str(logs_dir / f"{approach}.log"),
            "model_dir": str(out_dir),
        }
        if rc != 0:
            any_failed = True

    _finish(report, run_dir, success=not any_failed)
    sys.exit(1 if any_failed else 0)


def _finish(report: dict, run_dir: Path, success: bool, reason: str = ""):
    report["finished_at"] = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report["success"] = success
    if reason:
        report["failure_reason"] = reason

    json_path = run_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    md_lines = [
        f"# Classifier run report — {report['run_name']}",
        "",
        f"- Test mode: **{report['test_mode']}**",
        f"- Started: {report['started_at']}  Finished: {report['finished_at']}",
        f"- Overall result: {'✅ SUCCESS' if success else '❌ FAILED — ' + reason}",
        "",
        "## Pipeline steps",
        "",
    ]
    for step, info in report["steps"].items():
        if step == "test_subset":
            md_lines.append(f"- `{step}`: sampled {info.get('actual')}/{info.get('requested')} videos "
                             f"-> `{info.get('labels_path')}`")
        elif info.get("skipped"):
            md_lines.append(f"- `{step}`: skipped")
        else:
            ok = "✅" if info.get("returncode") == 0 else "❌"
            md_lines.append(f"- `{step}`: {ok} ({info.get('elapsed_sec')}s) — log: `{info.get('log')}`")

    md_lines += ["", "## Approaches", ""]
    for approach, info in report["approaches"].items():
        ok = "✅" if info.get("returncode") == 0 else "❌"
        md_lines.append(f"- **{approach}**: {ok} ({info.get('elapsed_sec')}s)")
        md_lines.append(f"  - model dir: `{info.get('model_dir')}`")
        md_lines.append(f"  - log: `{info.get('log')}`")

    if report["test_mode"]:
        md_lines += [
            "",
            "## Next step",
            "",
            ("This was a **test run** on a handful of videos to check that inputs/outputs "
             "work end-to-end. " + ("Everything passed — rerun without `--test` on your full "
             "dataset." if success else "Something failed — check the logs above before "
             "scaling up to the full dataset.")),
        ]

    md_path = run_dir / "report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nReport written to {md_path} (and {json_path})")


if __name__ == "__main__":
    main()
