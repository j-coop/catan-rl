import os
import random
import re
from typing import List, Any

import torch
import numpy as np
from tianshou.data import Batch

from marl.env.tianshou.actor import MaskedActor
from marl.env.tianshou.heuristic_bot import HeuristicCatanPolicy


class _SnapshotActor:
    """Thin inference-only wrapper around a loaded MaskedActor checkpoint."""

    def __init__(self, actor: MaskedActor, path: str):
        self.actor = actor
        self.path = path

    @torch.no_grad()
    def select_action(self, obs_dict: dict) -> int:
        """obs_dict: {"observation": np.ndarray, "action_mask": np.ndarray}"""
        obs = {
            "observation": torch.tensor(
                obs_dict["observation"][None], dtype=torch.float32, device=next(self.actor.parameters()).device
            ),
            "action_mask": torch.tensor(
                obs_dict["action_mask"][None], dtype=torch.int8, device=next(self.actor.parameters()).device
            ),
        }
        logits, _ = self.actor(obs)
        return int(torch.argmax(logits, dim=-1).item())

    def __repr__(self):
        return f"SnapshotActor({os.path.basename(self.path)})"


class _HeuristicWrapper:
    """Wraps HeuristicCatanPolicy to match the select_action(obs_dict) interface."""

    def __init__(self, policy: HeuristicCatanPolicy):
        self.policy = policy

    def select_action(self, obs_dict: dict) -> int:
        batch = Batch(obs={
            "observation": obs_dict["observation"][None],
            "action_mask": obs_dict["action_mask"][None],
        })
        result = self.policy.forward(batch)
        return int(result.act[0])

    def __repr__(self):
        return f"HeuristicBot(level={self.policy.level})"


class HistoryPool:
    """
    Maintains a pool of opponent policies: historical model snapshots + heuristic bots.

    Snapshots are drawn from one or more checkpoint directories (Phase 1 and Phase 2
    checkpoints are treated equally). refresh() uses stratified sampling to guarantee
    opponents span the full training history, not just the most recent checkpoints.

    sample_opponents(n) draws each slot independently with a 50% chance of picking a
    heuristic bot and 50% chance of picking a model snapshot.
    """

    def __init__(
        self,
        checkpoint_dirs: List[str],
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        max_snapshots: int = 20,
        heuristic_levels: List[int] = None,
    ):
        self.checkpoint_dirs = checkpoint_dirs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.max_snapshots = max_snapshots

        levels = heuristic_levels if heuristic_levels is not None else [1, 2, 3]
        self.heuristics = [
            _HeuristicWrapper(HeuristicCatanPolicy(level=l)) for l in levels
        ]
        self.snapshots: List[_SnapshotActor] = []
        self._loaded_paths: set = set()

    @staticmethod
    def _stratified_sample(sorted_pairs: list, n: int) -> list:
        """
        Pick n entries from sorted_pairs by dividing into n equal buckets and
        choosing one randomly from each. Ensures coverage across the full range.
        """
        total = len(sorted_pairs)
        if total <= n:
            return list(sorted_pairs)
        return [
            random.choice(sorted_pairs[(i * total) // n : ((i + 1) * total) // n])
            for i in range(n)
        ]

    def refresh(self):
        """
        Scan all checkpoint_dirs for .pt files, apply stratified sampling across
        the full history, and load any new snapshots into memory.
        """
        all_pairs: List[tuple] = []  # (dir, fname)
        for d in self.checkpoint_dirs:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".pt"):
                    all_pairs.append((d, f))

        def _epoch(pair):
            m = re.search(r"(\d+)\.pt$", pair[1])
            return int(m.group(1)) if m else -1

        all_pairs.sort(key=_epoch)
        candidates = self._stratified_sample(all_pairs, self.max_snapshots)

        new_snapshots = []
        for (d, fname) in candidates:
            path = os.path.join(d, fname)
            if path in self._loaded_paths:
                existing = next((s for s in self.snapshots if s.path == path), None)
                if existing:
                    new_snapshots.append(existing)
                continue
            try:
                actor = MaskedActor(self.obs_dim, self.act_dim).to(self.device)
                ckpt = torch.load(path, map_location=self.device)
                actor_state = {
                    k.removeprefix("actor."): v
                    for k, v in ckpt["policy"].items()
                    if k.startswith("actor.")
                }
                actor.load_state_dict(actor_state)
                actor.eval()
                snap = _SnapshotActor(actor, path)
                new_snapshots.append(snap)
                self._loaded_paths.add(path)
                print(f"[HistoryPool] Loaded snapshot: {fname} from {d}")
            except Exception as e:
                print(f"[HistoryPool] Failed to load {fname}: {e}")

        self.snapshots = new_snapshots

    def sample_opponents(self, n: int = 3) -> List[Any]:
        """
        Return n policies. Each slot is filled independently:
        50% chance heuristic bot, 50% chance model snapshot.
        Falls back to heuristics only if no snapshots are loaded.
        """
        result = []
        for _ in range(n):
            if not self.snapshots or random.random() < 0.5:
                result.append(random.choice(self.heuristics))
            else:
                result.append(random.choice(self.snapshots))
        return result

    def pool_size(self) -> int:
        return len(self.heuristics) + len(self.snapshots)

    def __repr__(self):
        return (
            f"HistoryPool(heuristics={len(self.heuristics)}, "
            f"snapshots={len(self.snapshots)})"
        )
