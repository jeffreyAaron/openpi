"""Asynchronous off-policy learner for RLT Stage 2.

Implements the paper's "Update" protocol:
  * Policy updates are performed off-policy from the replay buffer (Algorithm 1).
  * Rollouts and learning run asynchronously to stay compute- and time-efficient.
  * Two critic updates per actor update.
  * Learning begins shortly after the warm-up phase.
  * High update-to-data ratio (UTD=5) measured per environment step.

A background thread samples mini-batches from :class:`ReplayBuffer` and applies
critic / actor updates while the main thread continues to collect rollouts. The
ratio of cumulative gradient updates to cumulative environment steps is kept
near ``utd_ratio`` by a simple debt-counter: the learner only steps when the
current (actor+critic) update count is below ``utd_ratio * env_steps``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable

import torch

from openpi.rlt.actor_critic import (
    GaussianActor,
    TwinQCritic,
    soft_update_target,
)
from openpi.rlt.replay_buffer import ReplayBuffer


logger = logging.getLogger(__name__)


CriticUpdateFn = Callable[
    [
        dict[str, torch.Tensor],
        TwinQCritic,
        GaussianActor,
        TwinQCritic,
        torch.optim.Optimizer,
        float,
        int,
    ],
    float,
]

ActorUpdateFn = Callable[
    [
        dict[str, torch.Tensor],
        GaussianActor,
        TwinQCritic,
        torch.optim.Optimizer,
        float,
        float,
    ],
    tuple[float, float],
]


class AsyncLearner:
    """Background-thread learner that enforces UTD and the 2:1 critic:actor cadence.

    Thread-safety model:
      * ``replay_buffer`` is thread-safe (see ``ReplayBuffer._lock``).
      * Actor parameters are read during rollouts (forward_mean) and written by
        the learner. ``actor_lock`` serializes these accesses; callers running
        actor inference in the rollout must acquire it. The critic + target
        critic are only touched inside the learner thread, so they don't need
        a lock.

    Lifecycle:
      1. Construct the learner (cheap — starts no thread).
      2. Call :meth:`start` once warm-up is over.
      3. Call :meth:`record_env_steps` from the rollout loop whenever new env
         transitions have been collected.
      4. Drain metrics per episode via :meth:`pop_metrics`.
      5. Call :meth:`stop` before exiting training.
    """

    def __init__(
        self,
        *,
        replay_buffer: ReplayBuffer,
        actor: GaussianActor,
        critic: TwinQCritic,
        target_critic: TwinQCritic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        actor_lock: threading.Lock,
        update_critic_fn: CriticUpdateFn,
        update_actor_fn: ActorUpdateFn,
        device: torch.device,
        batch_size: int,
        discount: float,
        rl_chunk_length: int,
        bc_reg_weight: float,
        ref_action_dropout: float,
        tau: float,
        utd_ratio: int,
        critic_updates_per_actor: int,
        idle_sleep: float = 5e-4,
        min_buffer_size: int | None = None,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.actor_lock = actor_lock
        self._update_critic_fn = update_critic_fn
        self._update_actor_fn = update_actor_fn
        self.device = device
        self.batch_size = batch_size
        self.discount = discount
        self.rl_chunk_length = rl_chunk_length
        self.bc_reg_weight = bc_reg_weight
        self.ref_action_dropout = ref_action_dropout
        self.tau = tau
        self.utd_ratio = int(utd_ratio)
        self.critic_updates_per_actor = max(1, int(critic_updates_per_actor))
        self.idle_sleep = idle_sleep
        # Allow the buffer to fill up to a few batches before we start updating
        # so the very first gradients are over at least moderately diverse data.
        self.min_buffer_size = (
            batch_size if min_buffer_size is None else max(batch_size, int(min_buffer_size))
        )

        # Shared counters; guarded by _counter_lock.
        self._counter_lock = threading.Lock()
        self._env_steps = 0  # Cumulative env steps observed (from rollout).
        self._critic_update_count = 0
        self._actor_update_count = 0

        # Metrics; guarded by _metrics_lock. Keep a bounded recent history so
        # logging doesn't starve the learner if the main loop stalls.
        self._metrics_lock = threading.Lock()
        self._critic_losses: deque[float] = deque(maxlen=10_000)
        self._actor_losses: deque[float] = deque(maxlen=10_000)
        self._bc_losses: deque[float] = deque(maxlen=10_000)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ---- Public interface -------------------------------------------------

    def start(self) -> None:
        """Spawn the learner thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="rlt-async-learner", daemon=True,
        )
        self._thread.start()
        logger.info(
            "AsyncLearner started: utd_ratio=%d, critic_updates_per_actor=%d, batch_size=%d",
            self.utd_ratio, self.critic_updates_per_actor, self.batch_size,
        )

    def stop(self, timeout: float = 30.0) -> None:
        """Signal the learner to exit and join the thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "AsyncLearner thread did not exit within %.1fs; checkpoint state may be "
                    "mid-update. Consider increasing the stop timeout.", timeout,
                )
            self._thread = None

    def record_env_steps(self, num_steps: int) -> None:
        """Increment the cumulative env-step counter (called from rollout)."""
        if num_steps <= 0:
            return
        with self._counter_lock:
            self._env_steps += int(num_steps)

    def pop_metrics(self) -> dict[str, object]:
        """Drain recent metrics for logging.

        Returns a dict with:
          - ``critic_losses``: list of losses since the last call
          - ``actor_losses``: list of losses since the last call
          - ``bc_losses``: list of losses since the last call
          - ``env_steps``: cumulative env steps recorded
          - ``critic_updates``: cumulative critic updates so far
          - ``actor_updates``: cumulative actor updates so far
          - ``achieved_utd``: (critic+actor)/max(env_steps,1)
        """
        with self._metrics_lock:
            critic_losses = list(self._critic_losses)
            actor_losses = list(self._actor_losses)
            bc_losses = list(self._bc_losses)
            self._critic_losses.clear()
            self._actor_losses.clear()
            self._bc_losses.clear()
        with self._counter_lock:
            env_steps = self._env_steps
            critic_updates = self._critic_update_count
            actor_updates = self._actor_update_count
        total_updates = critic_updates + actor_updates
        return {
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
            "bc_losses": bc_losses,
            "env_steps": env_steps,
            "critic_updates": critic_updates,
            "actor_updates": actor_updates,
            "achieved_utd": total_updates / max(1, env_steps),
        }

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- Main loop --------------------------------------------------------

    def _can_step_now(self) -> bool:
        """Return True iff total updates are below the UTD budget."""
        with self._counter_lock:
            env_steps = self._env_steps
            total_updates = self._critic_update_count + self._actor_update_count
        return total_updates < self.utd_ratio * env_steps

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                # Wait for buffer to be warm enough for the first batch.
                if self.replay_buffer.size < self.min_buffer_size:
                    if self._stop_event.wait(timeout=self.idle_sleep * 20):
                        break
                    continue

                # Respect the UTD debt: only step when behind the target ratio.
                if not self._can_step_now():
                    if self._stop_event.wait(timeout=self.idle_sleep):
                        break
                    continue

                batch = self.replay_buffer.sample(self.batch_size, self.device)

                # --- Critic update -----------------------------------------
                # Critic update reads actor.forward_mean inside compute_td_target.
                # Acquire actor_lock to guarantee a consistent parameter view.
                with self.actor_lock:
                    critic_loss = self._update_critic_fn(
                        batch,
                        self.critic,
                        self.actor,
                        self.target_critic,
                        self.critic_optimizer,
                        self.discount,
                        self.rl_chunk_length,
                    )
                with self._counter_lock:
                    self._critic_update_count += 1
                    do_actor = (
                        self._critic_update_count % self.critic_updates_per_actor == 0
                    )
                with self._metrics_lock:
                    self._critic_losses.append(critic_loss)

                # --- Actor update every N critic steps ---------------------
                if do_actor:
                    with self.actor_lock:
                        actor_loss, bc_loss = self._update_actor_fn(
                            batch,
                            self.actor,
                            self.critic,
                            self.actor_optimizer,
                            self.bc_reg_weight,
                            self.ref_action_dropout,
                        )
                    with self._counter_lock:
                        self._actor_update_count += 1
                    with self._metrics_lock:
                        self._actor_losses.append(actor_loss)
                        self._bc_losses.append(bc_loss)

                # --- Target soft update ------------------------------------
                soft_update_target(self.target_critic, self.critic, self.tau)

        except Exception:  # noqa: BLE001 — surface errors from the learner thread
            logger.exception("AsyncLearner crashed; stopping thread.")
            self._stop_event.set()
            raise
