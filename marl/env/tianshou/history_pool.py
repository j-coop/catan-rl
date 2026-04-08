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
    Call refresh() periodically to load new checkpoints from disk.
    Call sample_opponents(n) to get n random policies for one episode.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        max_snapshots: int = 20,
        heuristic_levels: List[int] = None,
    ):
        self.checkpoint_dir = checkpoint_dir
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

    def refresh(self):
        """Scan checkpoint_dir for new .pt files and load them (up to max_snapshots newest)."""
        if not os.path.isdir(self.checkpoint_dir):
            return

        all_pts = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.endswith(".pt")
        ]

        def _epoch(name):
            m = re.search(r"(\d+)\.pt$", name)
            return int(m.group(1)) if m else -1

        all_pts.sort(key=_epoch)
        candidates = all_pts[-self.max_snapshots:]

        new_snapshots = []
        for fname in candidates:
            path = os.path.join(self.checkpoint_dir, fname)
            if path in self._loaded_paths:
                # Keep already-loaded actor
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
                print(f"[HistoryPool] Loaded snapshot: {fname}")
            except Exception as e:
                print(f"[HistoryPool] Failed to load {fname}: {e}")

        self.snapshots = new_snapshots

    def sample_opponents(self, n: int = 3) -> List[Any]:
        """Return n policies sampled uniformly from the full pool."""
        pool = self.heuristics + self.snapshots
        return random.choices(pool, k=n)

    def pool_size(self) -> int:
        return len(self.heuristics) + len(self.snapshots)

    def __repr__(self):
        return (
            f"HistoryPool(heuristics={len(self.heuristics)}, "
            f"snapshots={len(self.snapshots)})"
        )
