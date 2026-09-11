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

Merge multiple source folders (e.g. one per dataset) before extraction:
    python orchestrator.py --raw_videos cataracts/ cholec80/ kinetics_subset/ --labels labels.csv --out_dir runs/
    (Filenames are kept as-is unless two sources have a clashing filename,
    in which case the later file is prefixed with its source folder name --
    watch the printed [merge] warnings and update labels.csv's video_id
    for any renamed file.)

Label videos by folder instead of hand-writing labels.csv:
    python orchestrator.py \
        --not_surgery kinetics_subset/ something_something/ \
        --other_surgery cholec80/ jigsaws/ \
        --cataract cataracts/ cataract101/ \
        --out_dir runs/
    (Every video in a --not_surgery folder is labeled is_surgery=0; every
    video in --other_surgery is is_surgery=1, surgery_type=other_surgery;
    every video in --cataract is is_surgery=1, surgery_type=cataract.
    Folders are merged into one raw_videos_merged/ dir and labels.csv is
    written automatically to the run's output dir -- do not pass --labels
    or --raw_videos alongside these three flags.)
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


def merge_video_sources(sources: list[Path], staging_dir: Path, extensions: list[str]) -> Path:
    """
    Symlink (falling back to copy, e.g. across filesystems) video files from
    one or more source directories into a single flat staging directory, so
    the rest of the pipeline can keep treating raw video input as one
    folder -- extract_frames.py itself is untouched.

    Filenames are preserved as-is so they keep matching the video_id values
    already written in labels.csv. If two sources contain a file with the
    same name, the later one is disambiguated by prefixing it with its
    source folder's name, and a warning is printed so you can check whether
    labels.csv needs updating for that file.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    seen_names: dict[str, Path] = {}
    collisions = []

    for src_dir in sources:
        src_dir = Path(src_dir)
        if not src_dir.is_dir():
            raise FileNotFoundError(f"--raw_videos source not found or not a directory: {src_dir}")
        files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in extensions)
        for f in files:
            target_name = f.name
            if target_name in seen_names:
                target_name = f"{src_dir.name}_{f.name}"
                collisions.append((f, seen_names[f.name], staging_dir / target_name))
            seen_names.setdefault(f.name, f)

            target = staging_dir / target_name
            if target.exists() or target.is_symlink():
                continue  # already staged (e.g. re-run over the same out_dir)
            try:
                target.symlink_to(f.resolve())
            except OSError:
                shutil.copy2(f, target)

    if collisions:
        print(f"[merge] {len(collisions)} filename collision(s) across sources -- renamed to disambiguate:")
        for original, first_seen, renamed in collisions:
            print(f"  [warn] {original} collides with {first_seen}; staged as {renamed.name}"
                  f" -- update labels.csv's video_id for this file if needed")

    n_staged = sum(1 for _ in staging_dir.iterdir())
    print(f"[merge] Staged {n_staged} videos from {len(sources)} source folder(s) into {staging_dir}")
    return staging_dir


CATEGORY_TO_LABELS = {
    "not_surgery": {"is_surgery": 0, "surgery_type": ""},
    "other_surgery": {"is_surgery": 1, "surgery_type": "other_surgery"},
    "cataract": {"is_surgery": 1, "surgery_type": "cataract"},
}


def merge_and_label_by_category(category_dirs: dict[str, list[Path]], staging_dir: Path,
                                 extensions: list[str]) -> tuple[Path, "pd.DataFrame"]:
    """
    Merge video folders grouped by category ("not_surgery", "other_surgery",
    "cataract") into one staging directory AND build a labels DataFrame at
    the same time, so you never have to hand-write labels.csv -- you just
    say which folders belong to which category and every video in that
    folder gets that label.

    Uses the same collision-handling as merge_video_sources: if two source
    folders (even across different categories) contain a file with the same
    name, the later one is renamed with its source folder as a prefix, and
    a warning is printed.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    seen_names: dict[str, Path] = {}
    collisions = []
    rows = []

    for category in ("not_surgery", "other_surgery", "cataract"):
        for src_dir in category_dirs.get(category, []):
            src_dir = Path(src_dir)
            if not src_dir.is_dir():
                raise FileNotFoundError(f"--{category} source not found or not a directory: {src_dir}")
            files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in extensions)
            for f in files:
                target_name = f.name
                if target_name in seen_names:
                    target_name = f"{src_dir.name}_{f.name}"
                    collisions.append((f, seen_names[f.name], staging_dir / target_name))
                seen_names.setdefault(f.name, f)

                target = staging_dir / target_name
                if not target.exists() and not target.is_symlink():
                    try:
                        target.symlink_to(f.resolve())
                    except OSError:
                        shutil.copy2(f, target)

                label = CATEGORY_TO_LABELS[category]
                rows.append({"video_id": target.stem, "is_surgery": label["is_surgery"],
                             "surgery_type": label["surgery_type"]})

    if collisions:
        print(f"[merge] {len(collisions)} filename collision(s) across sources -- renamed to disambiguate:")
        for original, first_seen, renamed in collisions:
            print(f"  [warn] {original} collides with {first_seen}; staged as {renamed.name}")

    labels_df = pd.DataFrame(rows, columns=["video_id", "is_surgery", "surgery_type"])
    n_cataract = int((labels_df["surgery_type"] == "cataract").sum())
    n_other = int((labels_df["surgery_type"] == "other_surgery").sum())
    n_not = int((labels_df["is_surgery"] == 0).sum())
    print(f"[merge] Staged {len(labels_df)} videos into {staging_dir} "
          f"(cataract={n_cataract}, other_surgery={n_other}, not_surgery={n_not})")

    return staging_dir, labels_df


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
    parser.add_argument("--raw_videos", nargs="+", help="One or more directories of raw video files (skip if using --embeddings or the category flags below). Multiple directories are merged into one before extraction.")
    parser.add_argument("--labels", help="Path to labels.csv. Required unless using --not_surgery/--other_surgery/--cataract, which auto-generate it.")
    parser.add_argument("--not_surgery", nargs="+", help="Folder(s) whose videos are all NOT surgery (is_surgery=0). Alternative to --raw_videos/--labels.")
    parser.add_argument("--other_surgery", nargs="+", help="Folder(s) whose videos are all non-cataract surgery (is_surgery=1, surgery_type=other_surgery). Alternative to --raw_videos/--labels.")
    parser.add_argument("--cataract", nargs="+", help="Folder(s) whose videos are all cataract surgery (is_surgery=1, surgery_type=cataract). Alternative to --raw_videos/--labels.")
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

    using_category_folders = bool(args.not_surgery or args.other_surgery or args.cataract)

    if using_category_folders and not (args.not_surgery and args.other_surgery and args.cataract):
        parser.error("--not_surgery, --other_surgery, and --cataract must all be provided together "
                     "(the classifier trains a 3-class model and needs examples of all three)")
    if using_category_folders and args.raw_videos:
        parser.error("Use either --raw_videos, or --not_surgery/--other_surgery/--cataract, not both")
    if using_category_folders and args.labels:
        parser.error("--labels is auto-generated from --not_surgery/--other_surgery/--cataract; omit --labels")
    if not using_category_folders and not args.embeddings and not args.raw_videos:
        parser.error("Provide either --raw_videos, --not_surgery/--other_surgery/--cataract (to extract+embed), or --embeddings (precomputed)")
    if not using_category_folders and not args.labels:
        parser.error("--labels is required unless using --not_surgery/--other_surgery/--cataract")

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

    labels_path = Path(args.labels) if args.labels else None
    raw_videos_path = None

    if using_category_folders:
        category_dirs = {
            "not_surgery": [Path(p) for p in (args.not_surgery or [])],
            "other_surgery": [Path(p) for p in (args.other_surgery or [])],
            "cataract": [Path(p) for p in (args.cataract or [])],
        }
        n_folders = sum(len(v) for v in category_dirs.values())
        print(f"\n[merge] Building labels.csv from {n_folders} category folder(s)...")
        raw_videos_path, labels_df = merge_and_label_by_category(
            category_dirs, run_dir / "raw_videos_merged", args.extensions
        )
        labels_path = run_dir / "labels.csv"
        labels_df.to_csv(labels_path, index=False)
        report["steps"]["merge_and_label"] = {
            "not_surgery_dirs": [str(p) for p in category_dirs["not_surgery"]],
            "other_surgery_dirs": [str(p) for p in category_dirs["other_surgery"]],
            "cataract_dirs": [str(p) for p in category_dirs["cataract"]],
            "merged_dir": str(raw_videos_path),
            "labels_path": str(labels_path),
            "n_videos": len(labels_df),
        }
        print(f"[merge] Wrote {labels_path} ({len(labels_df)} rows)")
    elif args.raw_videos:
        raw_sources = [Path(p) for p in args.raw_videos]
        if len(raw_sources) == 1:
            raw_videos_path = raw_sources[0]
        else:
            print(f"\n[merge] Combining {len(raw_sources)} source folders into one...")
            raw_videos_path = merge_video_sources(raw_sources, run_dir / "raw_videos_merged", args.extensions)
            report["steps"]["merge_sources"] = {
                "sources": [str(p) for p in raw_sources],
                "merged_dir": str(raw_videos_path),
            }

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
        elif step == "merge_sources":
            md_lines.append(f"- `{step}`: merged {len(info.get('sources', []))} source folder(s) "
                             f"-> `{info.get('merged_dir')}`")
        elif step == "merge_and_label":
            n_dirs = (len(info.get('not_surgery_dirs', [])) + len(info.get('other_surgery_dirs', []))
                      + len(info.get('cataract_dirs', [])))
            md_lines.append(f"- `{step}`: labeled {info.get('n_videos')} videos from {n_dirs} folder(s) "
                             f"-> `{info.get('merged_dir')}`, `{info.get('labels_path')}`")
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