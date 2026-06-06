"""
Evaluate a sample of base-training checkpoints against all three heuristic
bot levels and save results to JSON for later plotting.

Usage:
    python marl/tests/evaluate_checkpoints.py \
        --checkpoints-dir marl/env/tianshou/trained_models/checkpoints \
        --every-n 10 \
        --num-games 400 \
        --output results/checkpoint_winrates.json

The agent always occupies a single seat; position is rotated across all four
slots (num_games / 4 per rotation) so turn-order bias is averaged out.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from marl.env.tianshou.multi_agent_env import CatanEnv
from marl.env.tianshou.heuristic_bot import HeuristicCatanPolicy
from marl.tests.run_final_eval import load_actor, apply_action, make_trained_action

PLAYER_NAMES = ["Blue Player", "Purple Player", "Yellow Player", "Green Player"]


def make_heuristic_action(env: CatanEnv, agent_name: str, bot: HeuristicCatanPolicy) -> int:
    obs_dict = env.observe(agent_name)
    mask = obs_dict["action_mask"]
    obs_vec = obs_dict["observation"]
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) == 0:
        return 230
    if bot.level == 1:
        return int(np.random.choice(valid_indices))
    return int(bot._choose_heuristic_action(mask, valid_indices, bot.level, obs_vec))


def run_single_game(env: CatanEnv, trained_slot: int, actor, bot: HeuristicCatanPolicy) -> str:
    env.reset()
    env.step_counter = 0
    while not env.game.game_over:
        current = env.agent_selection
        slot = PLAYER_NAMES.index(current)
        if slot == trained_slot:
            action = make_trained_action(env, current, actor)
        else:
            action = make_heuristic_action(env, current, bot)
        action_type = apply_action(current, action, env)
        if action_type == "end_turn":
            env.agent_selection = env.game.current_player.name
            env.game.handle_dice_roll()
    return env.game.winner


def evaluate_checkpoint(model_path: str, bot_level: int, num_games: int, pbar: tqdm) -> float:
    """Return win rate for the agent in model_path against bot_level bots."""
    env = CatanEnv()
    actor = load_actor(env, model_path)
    bot = HeuristicCatanPolicy(level=bot_level)

    games_per_slot = num_games // 4
    remainder = num_games % 4
    wins = 0
    total = 0

    for slot in range(4):
        n = games_per_slot + (1 if slot < remainder else 0)
        trained_name = PLAYER_NAMES[slot]
        for _ in range(n):
            winner = run_single_game(env, slot, actor, bot)
            if winner == trained_name:
                wins += 1
            pbar.update(1)
        total += n

    return wins / total if total > 0 else 0.0


def select_checkpoints(files: list[str], every_n: int) -> list[str]:
    """Keep every Nth file from the sorted list (index 0, N, 2N, …, last)."""
    selected = files[::every_n]
    # always include the last checkpoint
    if files[-1] not in selected:
        selected = selected + [files[-1]]
    return selected


def extract_number(filename: str) -> int:
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints vs heuristic bots; save results to JSON."
    )
    parser.add_argument("--checkpoints-dir", type=str,
                        default="marl/env/tianshou/trained_models/checkpoints",
                        help="Directory containing .pt checkpoint files")
    parser.add_argument("--every-n", type=int, default=10,
                        help="Evaluate every Nth checkpoint (default: 10)")
    parser.add_argument("--num-games", type=int, default=400,
                        help="Games per checkpoint per bot level (split across 4 positions)")
    parser.add_argument("--bot-levels", type=str, default="1,2,3",
                        help="Comma-separated bot levels to evaluate against")
    parser.add_argument("--output", type=str, default="results/checkpoint_winrates.json",
                        help="Output JSON path")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    checkpoints_path = Path(args.checkpoints_dir)
    if not checkpoints_path.exists():
        checkpoints_path = ROOT.parent / args.checkpoints_dir
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"Checkpoints dir not found: {args.checkpoints_dir}")

    bot_levels = [int(x) for x in args.bot_levels.split(",")]

    all_files = sorted(
        [f for f in os.listdir(checkpoints_path) if f.endswith(".pt")],
        key=extract_number,
    )
    if not all_files:
        raise RuntimeError(f"No .pt files found in {checkpoints_path}")

    selected = select_checkpoints(all_files, args.every_n)
    print(f"Found {len(all_files)} checkpoints, evaluating {len(selected)} "
          f"(every {args.every_n}), {args.num_games} games × {len(bot_levels)} levels each.")

    total_games = len(selected) * len(bot_levels) * args.num_games
    results = []
    with tqdm(total=total_games, desc="Games", unit="game") as pbar:
        for filename in tqdm(selected, desc="Checkpoints", position=1, leave=False):
            checkpoint_num = extract_number(filename)
            full_path = str(checkpoints_path / filename)
            entry = {"checkpoint": checkpoint_num, "win_rates": {}}
            for level in bot_levels:
                pbar.set_postfix(checkpoint=checkpoint_num, level=level)
                wr = evaluate_checkpoint(full_path, level, args.num_games, pbar)
                entry["win_rates"][str(level)] = round(wr, 4)
            results.append(entry)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "checkpoints_dir": str(checkpoints_path),
            "every_n": args.every_n,
            "num_games": args.num_games,
            "bot_levels": bot_levels,
            "num_checkpoints_evaluated": len(selected),
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
