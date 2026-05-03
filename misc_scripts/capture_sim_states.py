#!/usr/bin/env python3
"""Capture per-tape-position MuJoCo sim states at X sim-seconds into a scripted demo.

For each unique (yellow_x, yellow_y, duct_x, duct_y) tape position found in the
dataset directory, runs the scripted handover action code and snapshots the full
MuJoCo sim state the moment sim time reaches --seconds X.

The captured states let the online RL training environment (RobosuiteRLTEnv) reset
directly to the mid-demo configuration, so the policy only needs to learn from
that point forward rather than from the beginning of the scene.

By default the Franka env is built **with wrist cameras** (same as Stage 2
``RobosuiteRLTEnv``) so ``model.xml`` and flattened state vectors match training.
Use ``--no-wrist-cameras`` only for faster capture; then run Stage 2 with
``--robosuite_no_wrist_cameras`` and the same ``--controller-cfg`` / contact
parameters you used here.

Output directory (default: <dataset-dir>_states_<X>s/ next to dataset dir):
  index.json  -- [{yellow_x, yellow_y, duct_x, duct_y, state_idx}, ...]
  states.npz  -- 'states': [N, state_dim]  (flattened MjSimState per position)
  model.xml   -- MuJoCo model XML (identical geometry across all tape positions)

Usage:
  python misc_scripts/capture_sim_states.py \\
      --dataset-dir data/dataset_filtered_64_homing_removed \\
      --seconds 8 \\
      [--index path/to/handover_index.json] \\
      [--action-file path/to/action_code.py] \\
      [--output-dir sim_states_8s/] \\
      [--robosuite-root ~/robosuite]

  # Infer seconds from trim folder name (e.g. *_trim_8_20 -> 8s):
  python misc_scripts/capture_sim_states.py \\
      --dataset-dir data/dataset_filtered_64_homing_removed_trim_8_20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tape position parsing from folder names
# ---------------------------------------------------------------------------

def _parse_float_tokens(tokens: list[str]) -> list[float]:
    """Convert underscore-encoded float tokens into Python floats.

    Encoding rules (matches handover_executor dataset naming):
      '0_08'     -> 0.08   (integer part '_' decimal part)
      'neg0_065' -> -0.065 (leading 'neg' = negative sign, rest = integer part)
      '0_0'      -> 0.0
    """
    floats: list[float] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("neg"):
            int_part = t[3:]  # strip 'neg'
            dec_part = tokens[i + 1] if i + 1 < len(tokens) else "0"
            floats.append(-float(f"{int_part}.{dec_part}"))
            i += 2
        else:
            dec_part = tokens[i + 1] if i + 1 < len(tokens) else "0"
            floats.append(float(f"{t}.{dec_part}"))
            i += 2
    return floats


def parse_tape_position(dirname: str) -> tuple[float, float, float, float] | None:
    """Parse (yellow_x, yellow_y, duct_x, duct_y) from a handover dataset folder name.

    Handles both formats:
      handover_yellow_0_08_0_065_0_0_duct_0_08_neg0_065_0_0
      handover_yellow_0_08_0_065_0_0_duct_0_08_neg0_065_0_0_x_<...>_y_<...>_angle_<...>
    """
    s = dirname
    if s.startswith("handover_"):
        s = s[len("handover_"):]
    if not s.startswith("yellow_"):
        return None
    s = s[len("yellow_"):]

    parts = s.split("_duct_", 1)
    if len(parts) != 2:
        return None
    yellow_str, duct_str = parts

    # Strip optional perturbation suffix (_x_... or _angle_...)
    for marker in ("_x_", "_angle_"):
        if marker in duct_str:
            duct_str = duct_str.split(marker, 1)[0]
            break

    yellow_floats = _parse_float_tokens(yellow_str.split("_"))
    duct_floats = _parse_float_tokens(duct_str.split("_"))

    if len(yellow_floats) < 2 or len(duct_floats) < 2:
        logger.warning("Could not parse enough floats from %r", dirname)
        return None

    return yellow_floats[0], yellow_floats[1], duct_floats[0], duct_floats[1]


def collect_tape_positions_from_dir(
    dataset_dir: Path,
) -> list[tuple[float, float, float, float]]:
    """Scan dataset_dir for episode subdirectories and return unique tape positions."""
    seen: set[tuple[float, float, float, float]] = set()
    ordered: list[tuple[float, float, float, float]] = []

    for entry in sorted(dataset_dir.iterdir()):
        if not entry.is_dir():
            continue
        pos = parse_tape_position(entry.name)
        if pos is None:
            logger.debug("Skipping non-episode directory: %s", entry.name)
            continue
        if pos not in seen:
            seen.add(pos)
            ordered.append(pos)

    return ordered


def collect_tape_positions_from_json(
    json_path: Path,
) -> list[tuple[float, float, float, float]]:
    """Load tape positions from a handover_index.json (yellow_x/y, duct_x/y per entry)."""
    with json_path.open() as f:
        data = json.load(f)

    seen: set[tuple[float, float, float, float]] = set()
    ordered: list[tuple[float, float, float, float]] = []
    for entry in data:
        pos = (
            float(entry["yellow_x"]),
            float(entry["yellow_y"]),
            float(entry["duct_x"]),
            float(entry["duct_y"]),
        )
        if pos not in seen:
            seen.add(pos)
            ordered.append(pos)
    return ordered


# ---------------------------------------------------------------------------
# Infer --seconds from trim folder name
# ---------------------------------------------------------------------------

def infer_seconds_from_dirname(dirname: str) -> float | None:
    """Extract the start time from a folder name ending in _trim_<start>_<end>."""
    m = re.search(r"_trim_(\d+(?:[_\.]\d+)?)_\d+(?:[_\.]\d+)?$", dirname)
    if m:
        return float(m.group(1).replace("_", "."))
    return None


# ---------------------------------------------------------------------------
# Sim state capture
# ---------------------------------------------------------------------------

def capture_state_at_seconds(
    low_level_env,
    exec_env,
    action_code: str,
    target_seconds: float,
) -> dict | None:
    """Run action_code, snapshot sim state when sim.data.time >= target_seconds.

    Monkey-patches low_level_env.robosuite_env.sim.step to intercept the moment
    sim time crosses target_seconds. Returns {'state': np.ndarray, 'xml': str} or
    None if the action finishes before reaching target_seconds.
    """
    rs = low_level_env.robosuite_env
    captured: dict = {}
    orig_step = rs.sim.step
    orig_step2 = rs.sim.step2

    def _maybe_capture() -> None:
        if not captured and rs.sim.data.time >= target_seconds:
            captured["state"] = rs.sim.get_state().flatten()
            captured["xml"] = rs.sim.model.get_xml()

    def patched_step() -> None:
        orig_step()
        _maybe_capture()

    def patched_step2() -> None:
        orig_step2()
        _maybe_capture()

    # Robosuite defaults to lite_physics=True, which uses step1/step2 instead of step().
    rs.sim.step = patched_step
    rs.sim.step2 = patched_step2
    try:
        exec_env.step(action_code)
    finally:
        rs.sim.step = orig_step
        rs.sim.step2 = orig_step2

    if not captured:
        logger.warning(
            "Action code completed without reaching %.2f sim-seconds "
            "(demo was %.2f s long). State not captured.",
            target_seconds,
            rs.sim.data.time,
        )
        return None

    return captured


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture MuJoCo sim states at X seconds for each tape position.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir", type=Path, required=True,
        help="Dataset directory whose subfolders define the tape positions to capture.",
    )
    parser.add_argument(
        "--seconds", type=float, default=None,
        help=(
            "Sim time (seconds) at which to snapshot. "
            "If omitted, inferred from a _trim_<start>_<end> suffix in --dataset-dir name."
        ),
    )
    parser.add_argument(
        "--index", type=Path, default=None,
        help=(
            "Path to handover_index.json. Tape positions are read from its "
            "yellow_x/y, duct_x/y fields (cleaner than folder parsing). "
            "Falls back to parsing subfolder names when omitted."
        ),
    )
    parser.add_argument(
        "--action-file", type=Path, default=None,
        help=(
            "Scripted action Python file to run. "
            "Defaults to data_gen_scripts/actions/handover_action_code_default.py "
            "relative to --robosuite-root."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Directory to write index.json, states.npz, model.xml. "
            "Default: <dataset-dir>_states_<X>s/ (sibling of dataset dir)."
        ),
    )
    parser.add_argument(
        "--robosuite-root", type=Path,
        default=Path.home() / "robosuite",
        help="Root of the robosuite repo (contains api/, data_gen_scripts/, robosuite/). "
             "Default: ~/robosuite.",
    )
    parser.add_argument(
        "--controller-cfg", type=str,
        default="robosuite/environments/custom/configs/panda_joint_ctrl_slow.json",
        help="Robosuite composite controller JSON path (relative to --robosuite-root or absolute).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print which tape positions would be captured without running the sim.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only capture the first N tape positions (useful for a quick smoke test). "
        "Combine with --robosuite_fixed_layout_index N during Stage 2 to use the captured slot.",
    )
    parser.add_argument(
        "--no-wrist-cameras",
        action="store_true",
        help=(
            "Build the Franka env with agentview only (faster capture). "
            "Default is OFF: wrist cameras are ON so model.xml + state vector match "
            "openpi Stage 2 / RobosuiteRLTEnv (use_wrist_cameras=True). "
            "If you capture with this flag, run Stage 2 with --robosuite_use_wrist_cameras false "
            "and re-use the same controller JSON as --controller-cfg."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        logger.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    # Resolve --seconds
    target_seconds = args.seconds
    if target_seconds is None:
        target_seconds = infer_seconds_from_dirname(dataset_dir.name)
        if target_seconds is None:
            logger.error(
                "--seconds not provided and could not be inferred from directory name %r. "
                "Expected a suffix like _trim_8_20.",
                dataset_dir.name,
            )
            sys.exit(1)
        logger.info("Inferred target time from folder name: %.2f s", target_seconds)

    # Collect tape positions
    index_path = args.index
    if index_path is None:
        candidate = dataset_dir / "handover_index.json"
        if candidate.exists():
            index_path = candidate
            logger.info("Using handover_index.json found in dataset dir: %s", index_path)

    if index_path is not None:
        tape_positions = collect_tape_positions_from_json(index_path)
        logger.info("Loaded %d unique tape positions from %s", len(tape_positions), index_path)
    else:
        tape_positions = collect_tape_positions_from_dir(dataset_dir)
        logger.info(
            "Parsed %d unique tape positions from subfolder names in %s",
            len(tape_positions), dataset_dir,
        )

    if not tape_positions:
        logger.error("No tape positions found.")
        sys.exit(1)

    if args.limit is not None:
        if args.limit < 1:
            logger.error("--limit must be >= 1, got %d", args.limit)
            sys.exit(1)
        if args.limit < len(tape_positions):
            logger.info(
                "--limit %d: capturing first %d of %d tape positions.",
                args.limit, args.limit, len(tape_positions),
            )
            tape_positions = tape_positions[: args.limit]

    # Resolve output directory
    output_dir = args.output_dir
    if output_dir is None:
        x_str = str(target_seconds).rstrip("0").rstrip(".")
        output_dir = dataset_dir.parent / f"{dataset_dir.name}_states_{x_str}s"
    output_dir = output_dir.resolve()

    logger.info("Output directory: %s", output_dir)
    logger.info("Target sim time: %.2f s", target_seconds)
    logger.info("Tape positions to capture: %d", len(tape_positions))

    if args.dry_run:
        for i, (yx, yy, dx, dy) in enumerate(tape_positions):
            print(f"  [{i}] yellow=({yx:.4f}, {yy:.4f})  duct=({dx:.4f}, {dy:.4f})")
        print(f"\n[dry-run] Would write to: {output_dir}")
        return

    if args.no_wrist_cameras:
        logger.warning(
            "Capturing with --no-wrist-cameras (agentview only). For Stage 2 use "
            "--robosuite_no_wrist_cameras and ensure --controller-cfg matches."
        )

    # Set up sys.path for robosuite imports
    robosuite_root = args.robosuite_root.resolve()
    if not robosuite_root.exists():
        logger.error("--robosuite-root not found: %s", robosuite_root)
        sys.exit(1)

    # Prepend real pyroki package *before* robosuite_root; otherwise `import pyroki` can
    # resolve to an empty namespace from the sibling `pyroki/` directory at repo root.
    _pyroki_src: Path | None = None
    for rel in (
        Path("mike") / "dependencies" / "pyroki" / "src",
        Path("pyroki") / "src",
    ):
        cand = (robosuite_root / rel).resolve()
        init_py = cand / "pyroki" / "__init__.py"
        if init_py.is_file():
            _pyroki_src = cand
            break
    if _pyroki_src is None:
        logger.error(
            "Could not find pyroki under %s (tried mike/dependencies/pyroki/src and pyroki/src).",
            robosuite_root,
        )
        sys.exit(1)

    for p in (
        str(robosuite_root),
        str(robosuite_root / "mike" / "dependencies" / "robosuite"),
        str(_pyroki_src),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)

    import robosuite.macros as macros
    macros.IMAGE_CONVENTION = "opencv"

    # ``api/franka_priviledged_api.py`` does ``import open3d as o3d`` at module load,
    # but never uses the symbol. Skip the heavy dependency by registering a stub.
    if "open3d" not in sys.modules:
        import types as _types
        _o3d_stub = _types.ModuleType("open3d")
        _o3d_stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules["open3d"] = _o3d_stub

    from robosuite.environments.custom.franka_robosuite_tape_handover import (
        FrankaRobosuiteTapeHandover,
    )
    from robosuite.environments.custom.control.base_executor import (
        CodeExecutionEnvBase,
        CodeExecEnvConfig,
    )
    from api.franka_priviledged_api import FrankaControlTapeHandoverPrivilegedApi
    from api.base_api import register_api

    register_api(
        "franka-handover-privileged",
        lambda env: FrankaControlTapeHandoverPrivilegedApi(env),
    )

    # Resolve controller config
    ctrl_cfg = args.controller_cfg
    ctrl_path = Path(ctrl_cfg)
    if not ctrl_path.is_absolute():
        ctrl_path = robosuite_root / ctrl_cfg
    if not ctrl_path.exists():
        logger.error("Controller config not found: %s", ctrl_path)
        sys.exit(1)
    controller_cfg = str(ctrl_path)

    # Load action code
    if args.action_file is not None:
        action_file = args.action_file.resolve()
    else:
        action_file = (
            robosuite_root
            / "data_gen_scripts"
            / "actions"
            / "handover_action_code_default.py"
        )
    if not action_file.exists():
        logger.error("Action file not found: %s", action_file)
        sys.exit(1)
    action_code = action_file.read_text(encoding="utf-8").rstrip()
    logger.info("Loaded action code from: %s", action_file)

    # Capture loop
    model_xml_saved = False
    index_entries: list[dict] = []
    failed: list[tuple] = []
    captured_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (yellow_x, yellow_y, duct_x, duct_y) in enumerate(tape_positions):
        logger.info(
            "[%d/%d] Capturing: yellow=(%.4f, %.4f)  duct=(%.4f, %.4f)",
            i + 1, len(tape_positions), yellow_x, yellow_y, duct_x, duct_y,
        )

        # Stable filename derived from the tape position values.
        def _fmt(v: float) -> str:
            return f"neg{abs(v):.4f}".replace(".", "_") if v < 0 else f"{v:.4f}".replace(".", "_")
        state_filename = (
            f"yellow_{_fmt(yellow_x)}_{_fmt(yellow_y)}"
            f"_duct_{_fmt(duct_x)}_{_fmt(duct_y)}.npy"
        )

        try:
            low_level_env = FrankaRobosuiteTapeHandover(
                controller_cfg=controller_cfg,
                viser_debug=False,
                privileged=True,
                enable_render=False,
                use_wrist_cameras=not args.no_wrist_cameras,
                yellow_tape_offset=[yellow_x, yellow_y, 0.0],
                duct_tape_offset=[duct_x, duct_y, 0.0],
            )

            cfg = CodeExecEnvConfig(
                low_level=low_level_env,
                apis=["franka-handover-privileged"],
                prompt="Pick up the yellow tape with Arm 1 and hand it over to Arm 0.",
            )
            exec_env = CodeExecutionEnvBase(cfg)
            exec_env.reset()

            result = capture_state_at_seconds(
                low_level_env, exec_env, action_code, target_seconds,
            )

            if result is None:
                logger.warning("  SKIP — state not captured (demo too short?).")
                failed.append((yellow_x, yellow_y, duct_x, duct_y))
                continue

            # Save this position's state as its own .npy file immediately.
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / state_filename, result["state"])
            logger.info(
                "  OK — %s  (state_dim=%d, sim_time=%.3f s)",
                state_filename, len(result["state"]),
                low_level_env.robosuite_env.sim.data.time,
            )

            # Save model XML once (geometry is identical for all positions).
            if not model_xml_saved:
                (output_dir / "model.xml").write_text(result["xml"], encoding="utf-8")
                model_xml_saved = True

            index_entries.append({
                "yellow_x": yellow_x,
                "yellow_y": yellow_y,
                "duct_x": duct_x,
                "duct_y": duct_y,
                "state_file": state_filename,
            })
            captured_count += 1

            # Update index.json atomically after each capture.
            tmp_json = output_dir / "index.json.tmp"
            tmp_json.write_text(json.dumps(index_entries, indent=2), encoding="utf-8")
            tmp_json.replace(output_dir / "index.json")

        except Exception:
            logger.exception("  ERROR — exception during capture.")
            failed.append((yellow_x, yellow_y, duct_x, duct_y))

    if captured_count == 0:
        logger.error("No states captured. Aborting.")
        sys.exit(1)

    if failed:
        logger.warning(
            "%d position(s) failed to capture: %s",
            len(failed),
            [(f"yellow=({yx:.3f},{yy:.3f}) duct=({dx:.3f},{dy:.3f})")
             for yx, yy, dx, dy in failed],
        )

    logger.info(
        "Done. %d/%d positions captured → %s",
        captured_count, len(tape_positions), output_dir,
    )


if __name__ == "__main__":
    main()
