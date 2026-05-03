#!/usr/bin/env python3
"""
Trim all MP4 videos in the handover dataset from a start time to an end time.

Each episode directory contains three videos:
  - handover_agentview.mp4
  - handover_robot0_wrist.mp4
  - handover_robot1_wrist.mp4

Usage examples:
  # Trim all episodes from 1.0s to 8.0s (in-place, overwrite originals)
  python trim_videos.py --start 1.0 --end 8.0 --index path/to/handover_index.json

  # Use a custom dataset directory (episodes resolved relative to it)
  python trim_videos.py --start 1.0 --end 8.0 --index path/to/index.json --dataset-dir /path/to/dataset

  # Trim to a separate output directory (preserves originals)
  python trim_videos.py --start 1.0 --end 8.0 --index path/to/index.json --output-dir /path/to/trimmed

  # Trim only specific episode IDs
  python trim_videos.py --start 1.0 --end 8.0 --index path/to/index.json --ids 0 3 7

  # Preview what would happen without doing anything
  python trim_videos.py --start 1.0 --end 8.0 --index path/to/index.json --dry-run
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


ALL_VIDEO_FILES = [
    "handover_agentview.mp4",
    "handover_robot0_wrist.mp4",
    "handover_robot1_wrist.mp4",
]
AGENTVIEW_ONLY = ["handover_agentview.mp4"]

DATASET_DIR_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "dataset_with_randomization"


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found. Install it with: sudo apt install ffmpeg", file=sys.stderr)
        sys.exit(1)


def get_video_duration(path: Path) -> float | None:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def trim_video(src: Path, dst: Path, start: float, end: float) -> bool:
    """
    Trim src video from start to end seconds, write to dst.
    Uses output seeking (-ss after -i) for accurate frame-level start time,
    with -t (duration) so the end point is also exact.
    Returns True on success.
    """
    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",                        # overwrite output
        "-i", str(src),
        "-ss", str(start),           # output seek — accurate start time
        "-t", str(duration),         # duration from start
        "-c", "copy",                # stream copy — no re-encode
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ffmpeg error:\n{result.stderr[-800:]}", file=sys.stderr)
        return False
    return True


def load_index(index_path: Path) -> list[dict] | None:
    """Return parsed index, or None if the file doesn't exist."""
    if not index_path.exists():
        return None
    with open(index_path) as f:
        return json.load(f)


def index_from_dir(dataset_dir: Path) -> list[dict]:
    """Build a minimal index by scanning dataset_dir for subdirectories."""
    entries = []
    for i, ep_dir in enumerate(sorted(d for d in dataset_dir.iterdir() if d.is_dir())):
        entries.append({"id": i, "dir_name": ep_dir.name})
    return entries


def resolve_episode_dir(entry: dict, dataset_dir: Path) -> Path | None:
    ep_dir = dataset_dir / entry["dir_name"]
    return ep_dir if ep_dir.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim all MP4 videos in the handover dataset to [start, end] seconds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start", type=float, required=True,
        help="Start time in seconds (inclusive).",
    )
    parser.add_argument(
        "--end", type=float, required=True,
        help="End time in seconds (inclusive).",
    )
    parser.add_argument(
        "--index", type=Path, default=None,
        help=(
            "Path to the JSON index file. "
            "Defaults to handover_index.json inside --dataset-dir."
        ),
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DATASET_DIR_DEFAULT,
        help=(
            "Root directory containing the episode subdirectories. "
            f"Default: {DATASET_DIR_DEFAULT}"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Directory to write trimmed videos into (mirrors the episode structure). "
            "Defaults to a sibling folder named <dataset-dir>_trim_<start>_<end>."
        ),
    )
    parser.add_argument(
        "--ids", type=int, nargs="+", default=None,
        help="Episode IDs to process. If not set, all episodes are processed.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without actually trimming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    check_ffmpeg()

    if args.start < 0:
        print("ERROR: --start must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.end <= args.start:
        print("ERROR: --end must be greater than --start", file=sys.stderr)
        sys.exit(1)

    dataset_dir: Path = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    index_path = args.index.resolve() if args.index else dataset_dir / "handover_index.json"
    index = load_index(index_path)
    if index is None:
        print(f"No index file found at {index_path} — scanning {dataset_dir} directly (agentview only).")
        index = index_from_dir(dataset_dir)
        video_files = AGENTVIEW_ONLY
    else:
        video_files = ALL_VIDEO_FILES

    # Filter by IDs if requested
    if args.ids is not None:
        id_set = set(args.ids)
        index = [e for e in index if e["id"] in id_set]
        if not index:
            print("ERROR: no episodes matched the requested IDs.", file=sys.stderr)
            sys.exit(1)

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        start_str = str(args.start).rstrip("0").rstrip(".")
        end_str = str(args.end).rstrip("0").rstrip(".")
        output_dir = dataset_dir.parent / f"{dataset_dir.name}_trim_{start_str}_{end_str}"

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    total_videos = 0
    trimmed_ok = 0
    skipped = 0
    errors = 0

    for entry in index:
        ep_id = entry["id"]
        ep_dir = resolve_episode_dir(entry, dataset_dir)

        if ep_dir is None:
            print(f"[id={ep_id}] SKIP — episode directory not found: {entry['dir_name']}")
            skipped += 1
            continue

        print(f"[id={ep_id}] {entry['dir_name']}")

        dst_dir = output_dir / entry["dir_name"]

        # Trim target video files
        for video_name in video_files:
            src = ep_dir / video_name
            if not src.exists():
                print(f"  SKIP {video_name} — file not found")
                skipped += 1
                continue

            total_videos += 1
            dst = dst_dir / video_name

            if args.dry_run:
                duration = get_video_duration(src)
                dur_str = f"{duration:.2f}s" if duration is not None else "unknown duration"
                print(f"  [dry-run] trim  {video_name} ({dur_str}) → {dst}")
                trimmed_ok += 1
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)
            success = trim_video(src, dst, args.start, args.end)
            if success:
                print(f"  OK   trim  {video_name}")
                trimmed_ok += 1
            else:
                print(f"  ERR  trim  {video_name} — ffmpeg failed")
                errors += 1

        # Copy all other files (e.g. .npz joint data) unchanged
        for src in ep_dir.iterdir():
            if not src.is_file():
                continue
            if src.suffix.lower() in VIDEO_EXTENSIONS and src.name in video_files:
                continue  # already handled above
            dst = dst_dir / src.name
            if args.dry_run:
                print(f"  [dry-run] copy  {src.name}")
            else:
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  OK   copy  {src.name}")

    print()
    print(f"Done. {trimmed_ok}/{total_videos} videos trimmed, {skipped} skipped, {errors} errors.")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
