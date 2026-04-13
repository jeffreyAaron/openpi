"""
Convert robosuite `handover_step.py` rollout output to LeRobot format for openpi Stage 1 (TSH / pi05_tsh).

Expected input layout (per rollout directory), as written by
`robosuite/test_scripts/handover_step.py`:

  handover_agentview.mp4       -> exo_image
  handover_robot0_wrist.mp4    -> wrist_left_image or wrist_right_image (see --robot0-is-left)
  handover_robot1_wrist.mp4    -> the other wrist
  handover_robot0_joints.npz   -> joint_positions (N, 7), gripper_positions (N, 1 or 2)
  handover_robot1_joints.npz   -> same

State / actions follow `openpi.policies.tsh_policy.TSHInputs` (16D):

  [left_joint_0..6, left_gripper, right_joint_0..6, right_gripper]

There is no teleoperator command in sim data. Actions are set to the *next* absolute
state (same temporal shift as `tsh_to_lerobot.py` uses between yam and gello), so the
last timestep is dropped and length is N - 1.

Usage (single rollout folder):

  uv run scripts/dataset_conversion/handover_robosuite_to_lerobot.py \\
      --input /path/to/dataset/handover_run_dir \\
      --output_dir handover_lerobot \\
      --fps 30

Usage (parent folder containing many rollout subfolders):

  uv run scripts/dataset_conversion/handover_robosuite_to_lerobot.py \\
      --input /path/to/dataset \\
      --output_dir handover_lerobot \\
      --fps 30

Optional: write outside the Hugging Face cache:

  uv run scripts/dataset_conversion/handover_robosuite_to_lerobot.py \\
      --input ... --repo_id local/handover --output_root /data/lerobot_datasets

Then train with:

  uv run scripts/train_rlt_stage1.py ... --lerobot_repo_id /data/lerobot_datasets/local/handover
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro

from lerobot.common.datasets.compute_stats import compute_episode_stats
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# Match scripts/dataset_conversion/tsh_to_lerobot.py (openpi TSH config).
TARGET_H = 512
TARGET_W = 512
MJPEG_QUALITY = 3

DEFAULT_TASK_PROMPT = "Pick up the yellow tape with Arm 1 and hand it over to Arm 0."


def install_video(src: Path, dst: Path, fps: int) -> None:
    """Re-encode src as MJPEG, pad to TARGET_H x TARGET_W (black), CFR."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-r",
        str(fps),
        "-i",
        str(src),
        "-vf",
        f"pad={TARGET_W}:{TARGET_H}:0:0:black,setpts=N/(FRAME_RATE*TB)",
        "-vsync",
        "cfr",
        "-vcodec",
        "mjpeg",
        "-q:v",
        str(MJPEG_QUALITY),
        "-pix_fmt",
        "yuvj420p",
        "-video_track_timescale",
        str(fps),
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mjpeg encode failed for {src}:\n{result.stderr.decode(errors='replace')}"
        )


def get_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}:\n{result.stderr}")
    return int(result.stdout.strip())


def _gripper_scalar(g: np.ndarray) -> np.ndarray:
    """(N,) or (N,1) or (N,2+) -> (N,) float32."""
    g = np.asarray(g, dtype=np.float32)
    if g.ndim == 1:
        return g
    if g.ndim == 2:
        return np.mean(g, axis=-1)
    raise ValueError(f"Unexpected gripper shape {g.shape}")


def _stack_arm_state(joint_positions: np.ndarray, gripper_positions: np.ndarray) -> np.ndarray:
    """(N, 7) + gripper -> (N, 8)."""
    jp = np.asarray(joint_positions, dtype=np.float32)
    if jp.ndim != 2 or jp.shape[1] != 7:
        raise ValueError(f"Expected joint_positions (N, 7), got {jp.shape}")
    g = _gripper_scalar(gripper_positions)
    if g.shape[0] != jp.shape[0]:
        raise ValueError("gripper_positions length must match joint_positions")
    return np.concatenate([jp, g[:, None]], axis=1)


def build_combined_state(
    robot0: np.lib.npyio.NpzFile,
    robot1: np.lib.npyio.NpzFile,
    *,
    robot0_is_left: bool,
) -> np.ndarray:
    """Full 16D state trajectory (N, 16)."""
    s0 = _stack_arm_state(robot0["joint_positions"], robot0["gripper_positions"])
    s1 = _stack_arm_state(robot1["joint_positions"], robot1["gripper_positions"])
    if s0.shape[0] != s1.shape[0]:
        raise ValueError(
            f"robot0 length {s0.shape[0]} != robot1 length {s1.shape[0]}"
        )
    if robot0_is_left:
        left, right = s0, s1
    else:
        left, right = s1, s0
    return np.concatenate([left, right], axis=1)


def discover_rollout_dirs(input_root: Path) -> list[Path]:
    """One dir with files directly, or many subdirs each containing handover_agentview.mp4."""
    marker = "handover_agentview.mp4"
    if (input_root / marker).exists():
        return [input_root.resolve()]
    out: list[Path] = []
    for sub in sorted(input_root.iterdir()):
        if sub.is_dir() and (sub / marker).exists():
            out.append(sub.resolve())
    if not out:
        raise FileNotFoundError(
            f"No '{marker}' in {input_root} or its immediate subdirectories."
        )
    return out


def load_and_install_episode(
    rollout_dir: Path,
    episode_index: int,
    dataset: LeRobotDataset,
    *,
    fps: int,
    robot0_is_left: bool,
    base_stem: str = "handover",
) -> dict:
    """
    Load npz joint trajectories, align actions = next state, install three MP4s as MJPEG.
    """
    exo = rollout_dir / f"{base_stem}_agentview.mp4"
    w0 = rollout_dir / f"{base_stem}_robot0_wrist.mp4"
    w1 = rollout_dir / f"{base_stem}_robot1_wrist.mp4"
    j0 = rollout_dir / f"{base_stem}_robot0_joints.npz"
    j1 = rollout_dir / f"{base_stem}_robot1_joints.npz"
    for p in (exo, w0, w1, j0, j1):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    r0 = np.load(j0)
    r1 = np.load(j1)
    combined = build_combined_state(r0, r1, robot0_is_left=robot0_is_left)
    n = combined.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 timesteps in {rollout_dir}, got {n}")

    # Same convention as tsh_to_lerobot: state[t], action[t] = commanded/next absolute pose at t+1.
    states = combined[:-1].copy()
    actions = combined[1:].copy()
    n_steps = actions.shape[0]

    chunk_str = f"chunk-{episode_index // dataset.meta.chunks_size:03d}"
    ep_str = f"episode_{episode_index:06d}"
    video_root = dataset.root / "videos" / chunk_str

    paths = {
        "exo": exo,
        "wrist_left": w0 if robot0_is_left else w1,
        "wrist_right": w1 if robot0_is_left else w0,
    }
    install_video(paths["exo"], video_root / "exo_image" / f"{ep_str}.mp4", fps)
    install_video(
        paths["wrist_left"],
        video_root / "wrist_left_image" / f"{ep_str}.mp4",
        fps,
    )
    install_video(
        paths["wrist_right"],
        video_root / "wrist_right_image" / f"{ep_str}.mp4",
        fps,
    )

    video_n = get_frame_count(video_root / "exo_image" / f"{ep_str}.mp4")
    if video_n != n_steps:
        n_steps = min(video_n, n_steps)
        if n_steps < 1:
            raise RuntimeError(
                f"{rollout_dir}: after sync, n_steps < 1 (video {video_n}, joint-based {actions.shape[0]})"
            )
        actions = actions[:n_steps]
        states = states[:n_steps]
        print(
            f"WARNING: {rollout_dir.name}: video frames={video_n}, joint pairs={actions.shape[0]} -> using {n_steps}"
        )

    return {"actions": actions, "state": states, "n_steps": n_steps}


def save_episode_metadata(
    dataset: LeRobotDataset,
    ep_data: dict,
    episode_index: int,
    *,
    task_prompt: str,
) -> None:
    n_steps = ep_data["n_steps"]
    actions = ep_data["actions"]
    states = ep_data["state"]
    fps = dataset.fps

    frame_indices = np.arange(n_steps, dtype=np.int64)
    episode_indices = np.full(n_steps, episode_index, dtype=np.int64)
    timestamps = (frame_indices / fps).astype(np.float32)
    global_index = np.arange(
        dataset.meta.total_frames, dataset.meta.total_frames + n_steps, dtype=np.int64
    )

    task = task_prompt
    task_index = dataset.meta.get_task_index(task)
    if task_index is None:
        dataset.meta.add_task(task)
        task_index = dataset.meta.get_task_index(task)
    task_indices = np.full(n_steps, task_index, dtype=np.int64)

    parquet_path = dataset.root / dataset.meta.get_data_file_path(episode_index)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "index": pa.array(global_index, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "task_index": pa.array(task_indices, type=pa.int64()),
            "state": pa.array(states.tolist(), type=pa.list_(pa.float32(), 16)),
            "actions": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 16)),
            "prompt": pa.array([task] * n_steps, type=pa.string()),
        }
    )
    pq.write_table(table, parquet_path)

    ep_buffer = {
        "index": global_index,
        "episode_index": episode_indices,
        "frame_index": frame_indices,
        "timestamp": timestamps,
        "task_index": task_indices,
        "state": states,
        "actions": actions,
    }
    ep_stats = compute_episode_stats(ep_buffer, dataset.features)
    dataset.meta.save_episode(
        episode_index=episode_index,
        episode_length=n_steps,
        episode_tasks=[task],
        episode_stats=ep_stats,
    )


def main(
    input: Path,
    output_dir: str,
    repo_id: str | None = None,
    output_root: Path | None = None,
    fps: int = 30,
    robot0_is_left: bool = True,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    base_stem: str = "handover",
    overwrite: bool = True,
) -> None:
    """
    Args:
        input: Rollout directory (contains handover_*.mp4/npz) OR parent of many such dirs.
        output_dir: Directory name under HF_LEROBOT_HOME (or under output_root if set).
        repo_id: HuggingFace-style id stored in metadata (default: output_dir).
        output_root: If set, dataset root is output_root / output_dir (absolute local path).
        fps: Must match collection rate / ffmpeg input rate used when re-encoding (see handover_step --joint_state_fps;
            note the script currently saves MP4 with fps=20 in imageio — use 20 if you did not change that).
        robot0_is_left: If True, robot0 maps to TSH \"left\" (first 8 dims); robot1 to \"right\".
        base_stem: Filename stem for handover_step outputs (default \"handover\").
        overwrite: Remove existing output dataset directory before writing.
    """
    input = input.expanduser().resolve()
    rid = repo_id or output_dir
    root: Path | None
    if output_root is not None:
        root = Path(output_root).expanduser().resolve() / output_dir
    else:
        root = None

    # Same path LeRobotDatasetMetadata.create uses when root is None.
    out_default = HF_LEROBOT_HOME / rid
    out_path = root if root is not None else out_default

    if out_path.exists() and overwrite:
        shutil.rmtree(out_path)

    rollout_dirs = discover_rollout_dirs(input)
    total = len(rollout_dirs)

    dataset = LeRobotDataset.create(
        repo_id=rid,
        fps=fps,
        root=root,
        robot_type="franka",
        features={
            "exo_image": {
                "dtype": "video",
                "shape": (TARGET_H, TARGET_W, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_left_image": {
                "dtype": "video",
                "shape": (TARGET_H, TARGET_W, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_right_image": {
                "dtype": "video",
                "shape": (TARGET_H, TARGET_W, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (16,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (16,), "names": ["actions"]},
            "prompt": {"dtype": "string", "shape": (1,), "names": ["prompt"]},
        },
        image_writer_threads=0,
        image_writer_processes=0,
    )

    for ep_idx, rd in enumerate(
        tqdm.tqdm(rollout_dirs, desc="Converting handover rollouts", total=total)
    ):
        ep_data = load_and_install_episode(
            rd,
            ep_idx,
            dataset,
            fps=fps,
            robot0_is_left=robot0_is_left,
            base_stem=base_stem,
        )
        save_episode_metadata(
            dataset,
            ep_data,
            ep_idx,
            task_prompt=task_prompt,
        )

    print(f"Wrote LeRobot dataset to {dataset.root} ({total} episodes).")


if __name__ == "__main__":
    tyro.cli(main)
