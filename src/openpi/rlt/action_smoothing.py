"""Chunk-level action smoothing for rollout (matches groundTruthEval temporal / EMA overlap)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ActionSmoothingName = Literal["none", "temporal_ensembling", "ema_overlap"]

# Bimanual Franka TSH: GRIP targets in [-1, 1]. Blending grippers across overlapping
# chunks washes out sharp "close" commands (older chunks still say "open" during approach).
_TSH_GRIPPER_DIMS = (7, 15)


class TemporalEnsembler:
    """Weighted blend of overlapping action chunks (ACT-style temporal ensembling)."""

    def __init__(self, k: float = 0.1) -> None:
        self.k = k
        self._chunks: list[np.ndarray] = []
        self._ages: list[int] = []

    def add_chunk(self, chunk: np.ndarray) -> None:
        self._chunks.append(np.asarray(chunk, dtype=np.float32).copy())
        self._ages.append(0)

    def get_action(self) -> np.ndarray:
        actions, weights = [], []
        for chunk, age in zip(self._chunks, self._ages):
            if age < len(chunk):
                actions.append(chunk[age])
                weights.append(float(np.exp(-self.k * age)))
        w = np.array(weights, dtype=np.float64)
        w /= w.sum()
        blended = np.sum([wi * ai for wi, ai in zip(w, actions)], axis=0).astype(np.float32)
        # Do not average gripper with stale chunks — keeps close commands from being diluted.
        if self._chunks and self._ages:
            newest_i = int(np.argmin(self._ages))
            age_n = self._ages[newest_i]
            ch = self._chunks[newest_i]
            if age_n < len(ch):
                for g in _TSH_GRIPPER_DIMS:
                    blended[g] = float(ch[age_n, g])
        return blended

    def step(self) -> None:
        self._ages = [a + 1 for a in self._ages]
        pairs = [(c, a) for c, a in zip(self._chunks, self._ages) if a < len(c)]
        if pairs:
            self._chunks, self._ages = map(list, zip(*pairs))
        else:
            self._chunks, self._ages = [], []

    def reset(self) -> None:
        self._chunks, self._ages = [], []


@dataclass
class EMAOverlapState:
    """Overlap EMA between consecutive chunks (groundTruthEval non-TE path)."""

    ema_alpha: float = 0.75
    curr_chunk: np.ndarray | None = None
    action_idx: int = 0

    def reset(self) -> None:
        self.curr_chunk = None
        self.action_idx = 0

    def ingest_new_chunk(self, new_chunk: np.ndarray, chunk_size: int) -> None:
        new_chunk = np.asarray(new_chunk, dtype=np.float32).copy()
        if (
            self.curr_chunk is not None
            and self.action_idx < chunk_size
            and len(self.curr_chunk) >= chunk_size
        ):
            overlap = chunk_size - self.action_idx
            gidx = list(_TSH_GRIPPER_DIMS)
            grip_saved = new_chunk[:overlap, gidx].copy()
            new_chunk[:overlap] = (
                self.ema_alpha * new_chunk[:overlap]
                + (1.0 - self.ema_alpha)
                * self.curr_chunk[self.action_idx : self.action_idx + overlap]
            )
            new_chunk[:overlap, gidx] = grip_saved
        self.curr_chunk = new_chunk
        self.action_idx = 0

    def next_action(self) -> np.ndarray:
        if self.curr_chunk is None:
            raise RuntimeError("EMAOverlapState: no chunk; call ingest_new_chunk first.")
        a = self.curr_chunk[self.action_idx].copy()
        self.action_idx += 1
        return a


@dataclass
class ActionSmoothingRuntime:
    """Per-episode state for optional smoothing."""

    mode: ActionSmoothingName
    te_k: float
    ema_alpha: float
    temporal: TemporalEnsembler | None = None
    ema: EMAOverlapState | None = None

    @classmethod
    def start_episode(
        cls,
        mode: str,
        *,
        te_k: float = 0.1,
        ema_alpha: float = 0.75,
    ) -> ActionSmoothingRuntime:
        if mode == "temporal_ensembling":
            return cls(
                mode="temporal_ensembling",
                te_k=te_k,
                ema_alpha=ema_alpha,
                temporal=TemporalEnsembler(k=te_k),
            )
        if mode == "ema_overlap":
            return cls(
                mode="ema_overlap",
                te_k=te_k,
                ema_alpha=ema_alpha,
                ema=EMAOverlapState(ema_alpha=ema_alpha),
            )
        if mode == "none":
            return cls(mode="none", te_k=te_k, ema_alpha=ema_alpha)
        raise ValueError(
            f"Unknown action_smoothing {mode!r}; expected none, temporal_ensembling, or ema_overlap."
        )

    def reset(self) -> None:
        if self.temporal is not None:
            self.temporal.reset()
        if self.ema is not None:
            self.ema.reset()

    def on_new_chunk(self, chunk: np.ndarray, chunk_size: int) -> None:
        if self.mode == "temporal_ensembling":
            assert self.temporal is not None
            self.temporal.add_chunk(chunk)
        elif self.mode == "ema_overlap":
            assert self.ema is not None
            self.ema.ingest_new_chunk(chunk, chunk_size)

    def next_executable_action(self, chunk: np.ndarray, step_in_cycle: int) -> np.ndarray:
        """Return the action to send to the environment for this timestep."""
        if self.mode == "temporal_ensembling":
            assert self.temporal is not None
            a = self.temporal.get_action()
            self.temporal.step()
            return a.astype(np.float32, copy=False)
        if self.mode == "ema_overlap":
            assert self.ema is not None
            return self.ema.next_action()
        return np.asarray(chunk[step_in_cycle], dtype=np.float32)
