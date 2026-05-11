"""GTE-style parallel Robosuite rollouts for RLT stage 2.

Subprocess workers each run :class:`RobosuiteRLTEnv` with EGL pinned per worker
(``CUDA_VISIBLE_DEVICES`` / ``MUJOCO_EGL_DEVICE_ID``), mirroring
``groundTruthEval.sim_worker``. The training process batches VLA + RL-token +
actor inference for all envs that are waiting at an inference boundary.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue
from queue import Empty
from typing import Any

import numpy as np

from openpi.rlt.shaped_reward import ShapedRewardConfig

logger = logging.getLogger(__name__)


def visible_gpu_ids_from_env() -> list[int]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible is None or cuda_visible.strip() == "":
        import torch

        return list(range(torch.cuda.device_count()))
    return [int(x.strip()) for x in cuda_visible.split(",") if x.strip() != ""]


@dataclass
class RobosuiteWorkerSpec:
    """Picklable config for a rollout worker (no torch / Policy objects)."""

    controller_cfg: str
    image_size: int | None
    max_steps: int
    base_seed: int
    tape_offsets_json: str | None
    contact_solref: list[float] | None
    contact_solimp: list[float] | None
    tape_layout_index: int | None
    gripper_action_log: str | None
    z_rl_dim: int
    # Shaped reward config carried as a plain dict for picklability across the
    # multiprocessing boundary; ``None`` keeps the legacy sparse 0/1 reward.
    shaped_reward_cfg: dict | None = None


def _action_queue_get(action_queue: Queue, stop_flag: Event, *, timeout: float = 0.5) -> dict[str, Any] | None:
    """Blocking get with shutdown support (avoids deadlock on stop)."""
    while True:
        try:
            return action_queue.get(timeout=timeout)
        except Empty:
            if stop_flag.is_set():
                return None


def robosuite_rlt_rollout_worker(
    env_id: int,
    spec: RobosuiteWorkerSpec,
    obs_queue: Queue,
    action_queue: Queue,
    transition_queue: Queue,
    stop_flag: Event,
    visible_gpu_ids: list[int],
    infer_every: int,
    rl_chunk_length: int,
    action_dim: int,
    action_smoothing: str,
    te_k: float,
    ema_alpha: float,
) -> None:
    """One Robosuite env loop; requests batched inference via ``obs_queue`` / ``action_queue``."""
    from openpi.rlt.action_smoothing import ActionSmoothingRuntime
    from openpi.rlt.robosuite_env import RobosuiteRLTEnv

    if not visible_gpu_ids:
        gpu_id = 0
    else:
        gpu_id = int(visible_gpu_ids[env_id % len(visible_gpu_ids)])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    os.environ["EGL_DEVICE_ID"] = str(gpu_id)

    shaped_cfg = (
        ShapedRewardConfig(**spec.shaped_reward_cfg)
        if spec.shaped_reward_cfg is not None
        else None
    )
    env = RobosuiteRLTEnv(
        controller_cfg=spec.controller_cfg,
        image_size=spec.image_size,
        max_steps=spec.max_steps,
        seed=spec.base_seed + env_id,
        tape_offsets_json=spec.tape_offsets_json,
        contact_solref=spec.contact_solref,
        contact_solimp=spec.contact_solimp,
        tape_layout_index=spec.tape_layout_index,
        gripper_action_log=spec.gripper_action_log if env_id == 0 else None,
        shaped_reward_cfg=shaped_cfg,
    )

    C, d = rl_chunk_length, action_dim
    z_rl_np = np.zeros((spec.z_rl_dim,), dtype=np.float32)
    ref_chunk_np = np.zeros((C, d), dtype=np.float32)
    action_chunk = np.zeros((C, d), dtype=np.float32)

    while not stop_flag.is_set():
        obs_dict = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        smooth = ActionSmoothingRuntime.start_episode(
            action_smoothing,
            te_k=te_k,
            ema_alpha=ema_alpha,
        )
        new_episode = True
        episode_serial = -1
        collect_video = False
        video_frames: list[np.ndarray] = []

        while not done and ep_steps < spec.max_steps:
            if stop_flag.is_set():
                return

            if ep_steps % infer_every == 0:
                obs_queue.put(
                    {
                        "env_id": env_id,
                        "obs_dict": obs_dict,
                        "new_episode": new_episode,
                        "want_video_frame": env_id == 0,
                    }
                )
                new_episode = False
                resp = _action_queue_get(action_queue, stop_flag)
                if resp is None:
                    return
                if resp.get("shutdown"):
                    return
                episode_serial = int(resp["episode_serial"])
                collect_video = bool(resp.get("collect_video", False))
                z_rl_np = np.asarray(resp["z_rl"], dtype=np.float32).reshape(-1)
                ref_chunk_np = np.asarray(resp["ref_chunk"], dtype=np.float32).reshape(C, d)
                action_chunk = np.asarray(resp["action_chunk"], dtype=np.float32).reshape(C, d)
                smooth.on_new_chunk(action_chunk, C)

            if collect_video and ep_steps % infer_every == 0 and "exo_image" in obs_dict:
                video_frames.append(np.asarray(obs_dict["exo_image"]).copy())

            step_in_cycle = ep_steps % infer_every
            state_before = np.asarray(obs_dict["state"], dtype=np.float32).reshape(-1)
            a_exec = smooth.next_executable_action(action_chunk, step_in_cycle)
            ref_row = np.asarray(ref_chunk_np[step_in_cycle], dtype=np.float32).reshape(-1)

            obs_dict, reward, done, info = env.step(a_exec)
            ep_reward += float(reward)
            ep_steps += 1

            transition_queue.put(
                {
                    "env_id": env_id,
                    "episode_serial": episode_serial,
                    "z_rl": z_rl_np.copy(),
                    "state": state_before,
                    "a_exec": np.asarray(a_exec, dtype=np.float32).reshape(-1),
                    "ref_row": ref_row,
                    "reward": float(reward),
                    "done": bool(done),
                    "ep_reward": ep_reward,
                    "ep_steps": ep_steps,
                    "success": bool(env.task_completed()) if done else False,
                    # Phase fields are always present (zeros when shaping is off);
                    # downstream consumers can read them unconditionally.
                    "phase": int(info.get("phase", 0)),
                    "phase_max": int(info.get("phase_max", 0)),
                    "shaping_reward": float(info.get("shaping_reward", 0.0)),
                    "milestone_reward": float(info.get("milestone_reward", 0.0)),
                    "video_frames": list(video_frames) if (done and env_id == 0) else None,
                }
            )
            if done and env_id == 0:
                video_frames = []


class ParallelRobosuiteRolloutPool:
    """Owns worker processes and queues."""

    def __init__(
        self,
        num_envs: int,
        spec: RobosuiteWorkerSpec,
        visible_gpu_ids: list[int],
        infer_every: int,
        rl_chunk_length: int,
        action_dim: int,
        action_smoothing: str,
        te_k: float,
        ema_alpha: float,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = num_envs
        self.obs_queue: Queue = Queue()
        self.transition_queue: Queue = Queue()
        self.action_queues: list[Queue] = [Queue() for _ in range(num_envs)]
        self.stop_flag = Event()
        self._processes: list[Process] = []
        for i in range(num_envs):
            p = Process(
                target=robosuite_rlt_rollout_worker,
                args=(
                    i,
                    spec,
                    self.obs_queue,
                    self.action_queues[i],
                    self.transition_queue,
                    self.stop_flag,
                    visible_gpu_ids,
                    infer_every,
                    rl_chunk_length,
                    action_dim,
                    action_smoothing,
                    te_k,
                    ema_alpha,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

    def _drain_queues_for_shutdown(self, *, rounds: int = 80, pause_s: float = 0.05) -> None:
        """Empty IPC queues so workers never stay blocked on ``put`` (pipe backpressure).

        When training stops, the main loop no longer consumes ``obs_queue`` /
        ``transition_queue``. Workers can then block forever on ``put``, and
        ``join`` times out even though sending ``shutdown`` on ``action_queue``
        is correct for workers already waiting on ``get``.
        """
        for _ in range(rounds):
            drained = False
            while True:
                try:
                    self.obs_queue.get_nowait()
                    drained = True
                except Empty:
                    break
            while True:
                try:
                    self.transition_queue.get_nowait()
                    drained = True
                except Empty:
                    break
            if not drained:
                time.sleep(pause_s)

    def close(self) -> None:
        self.stop_flag.set()
        for q in self.action_queues:
            q.put({"shutdown": True})
        # Unblock workers stuck on obs_queue.put / transition_queue.put.
        self._drain_queues_for_shutdown()
        for p in self._processes:
            p.join(timeout=120.0)
            if p.is_alive():
                logger.warning("Worker %s did not exit cleanly; terminating.", p.pid)
                p.terminate()
                p.join(timeout=10.0)

    def pending_obs_count(self) -> int:
        """Best-effort size (not reliable on all platforms); prefer draining."""
        try:
            return int(self.obs_queue.qsize())
        except NotImplementedError:
            return 0
