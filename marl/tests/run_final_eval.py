import argparse
import csv
import json
import os
import random
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
from marl.tests.agent_vs_random import load_actor, apply_action

PLAYER_NAMES = ["Blue Player", "Purple Player", "Yellow Player", "Green Player"]

DEFAULT_BASE_MODEL = ROOT / "trained_models" / "pretrained_model.pt"
DEFAULT_FT_MODEL = ROOT / "trained_models" / "ft_model.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Final evaluation: trained agents vs heuristic bots across all thesis configurations."
    )
    p.add_argument("--base-model", type=str, default=str(DEFAULT_BASE_MODEL),
                   help="Path to base model checkpoint (.pt)")
    p.add_argument("--ft-model", type=str, default=str(DEFAULT_FT_MODEL),
                   help="Path to fine-tuned model checkpoint (.pt)")
    p.add_argument("--model", choices=["base", "ft", "both"], default="both",
                   help="Which model variant(s) to evaluate")
    p.add_argument("--num-games", type=int, default=1000,
                   help="Total games per configuration (will be split equally across 4 starting positions)")
    p.add_argument("--bot-levels", type=str, default="1,2,3",
                   help="Comma-separated heuristic bot levels to test")
    p.add_argument("--agent-counts", type=str, default="1,2,3",
                   help="Comma-separated number of trained agents per game (1, 2, or 3)")
    p.add_argument("--output-dir", type=str, default="evaluation/results",
                   help="Directory where JSON and CSV results are saved")
    p.add_argument("--seed", type=int, default=None, help="Global random seed for reproducibility")
    return p.parse_args()


def compute_trained_positions(num_trained: int, rotation: int) -> list[int]:
    """Cyclic rotation of trained-agent slot indices for position-bias control."""
    return sorted([(i + rotation) % 4 for i in range(num_trained)])


def make_trained_action(env: CatanEnv, agent_name: str, actor) -> int:
    obs_dict = env.observe(agent_name)
    obs_vec = obs_dict["observation"][None, :]
    mask = [obs_dict["action_mask"]]
    obs = {"observation": obs_vec, "action_mask": mask}
    with torch.no_grad():
        logits, _ = actor(obs)
    return int(torch.argmax(logits).item())


def make_heuristic_action(env: CatanEnv, agent_name: str, bot: HeuristicCatanPolicy) -> int:
    obs_dict = env.observe(agent_name)
    mask = obs_dict["action_mask"]
    obs_vec = obs_dict["observation"]
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) == 0:
        return 230  # fallback: end turn
    if bot.level == 1:
        return int(np.random.choice(valid_indices))
    return int(bot._choose_heuristic_action(mask, valid_indices, bot.level, obs_vec))


def run_single_game(
    env: CatanEnv,
    trained_positions: list[int],
    actor,
    bot: HeuristicCatanPolicy,
) -> dict:
    env.reset()
    env.step_counter = 0
    while not env.game.game_over:
        current = env.agent_selection
        slot = PLAYER_NAMES.index(current)
        if slot in trained_positions:
            action = make_trained_action(env, current, actor)
        else:
            action = make_heuristic_action(env, current, bot)
        action_type = apply_action(current, action, env)
        if action_type == "end_turn":
            env.agent_selection = env.game.current_player.name
            env.game.handle_dice_roll()
    return {
        "winner": env.game.winner,
        "vps": {p.name: p.victory_points for p in env.game.players},
    }


def run_configuration(
    model_tag: str,
    model_path: str,
    bot_level: int,
    num_trained: int,
    num_games: int,
    seed: int | None,
) -> dict:
    env = CatanEnv()
    actor = load_actor(env, model_path)
    bot = HeuristicCatanPolicy(level=bot_level)

    bot_positions_set = set(range(4)) - set(range(num_trained))  # placeholder, updated per rotation

    games_per_rotation = num_games // 4
    remainder = num_games % 4

    total_trained_wins = 0
    total_trained_vp_sum = 0.0
    total_bot_vp_sum = 0.0
    total_games_played = 0

    per_rotation = []

    desc = f"[{model_tag}] L{bot_level} ×{num_trained}"
    with tqdm(total=num_games, desc=desc, unit="game", leave=False) as pbar:
        for rotation in range(4):
            trained_pos = compute_trained_positions(num_trained, rotation)
            bot_pos = [i for i in range(4) if i not in trained_pos]
            trained_names = [PLAYER_NAMES[p] for p in trained_pos]

            n_games = games_per_rotation + (1 if rotation < remainder else 0)

            rot_trained_wins = 0
            rot_trained_vp_sum = 0.0
            rot_bot_vp_sum = 0.0

            for _ in range(n_games):
                result = run_single_game(env, trained_pos, actor, bot)
                winner = result["winner"]
                vps = result["vps"]

                if winner in trained_names:
                    rot_trained_wins += 1
                    total_trained_wins += 1

                for slot in trained_pos:
                    rot_trained_vp_sum += vps[PLAYER_NAMES[slot]]
                    total_trained_vp_sum += vps[PLAYER_NAMES[slot]]
                for slot in bot_pos:
                    rot_bot_vp_sum += vps[PLAYER_NAMES[slot]]
                    total_bot_vp_sum += vps[PLAYER_NAMES[slot]]

                pbar.update(1)

            total_games_played += n_games
            rot_trained_count = n_games * num_trained
            rot_bot_count = n_games * (4 - num_trained)

            per_rotation.append({
                "rotation": rotation,
                "trained_slots": trained_pos,
                "trained_names": trained_names,
                "games": n_games,
                "trained_wins": rot_trained_wins,
                "win_rate": rot_trained_wins / n_games if n_games > 0 else 0.0,
                "avg_vp_trained": rot_trained_vp_sum / rot_trained_count if rot_trained_count > 0 else 0.0,
                "avg_vp_bot": rot_bot_vp_sum / rot_bot_count if rot_bot_count > 0 else 0.0,
            })

    trained_vp_count = total_games_played * num_trained
    bot_vp_count = total_games_played * (4 - num_trained)

    return {
        "model": model_tag,
        "model_path": str(model_path),
        "bot_level": bot_level,
        "num_trained": num_trained,
        "num_games": total_games_played,
        "trained_wins": total_trained_wins,
        "win_rate": total_trained_wins / total_games_played if total_games_played > 0 else 0.0,
        "avg_vp_trained": total_trained_vp_sum / trained_vp_count if trained_vp_count > 0 else 0.0,
        "avg_vp_bot": total_bot_vp_sum / bot_vp_count if bot_vp_count > 0 else 0.0,
        "per_rotation": per_rotation,
    }


def format_table(results: list[dict]) -> str:
    header = (
        f"{'model':<8} | {'bot_lvl':>7} | {'n_agents':>8} | {'games':>6} | "
        f"{'win_rate%':>9} | {'avg_vp_trained':>14} | {'avg_vp_bot':>10}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        rows.append(
            f"{r['model']:<8} | {r['bot_level']:>7} | {r['num_trained']:>8} | {r['num_games']:>6} | "
            f"{r['win_rate']*100:>8.2f}% | {r['avg_vp_trained']:>14.2f} | {r['avg_vp_bot']:>10.2f}"
        )
    return "\n".join(rows)


def save_json(results: list[dict], output_dir: Path, timestamp: str, metadata: dict):
    path = output_dir / f"final_eval_{timestamp}.json"
    payload = {"metadata": metadata, "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON  → {path}")


def save_csv(results: list[dict], output_dir: Path, timestamp: str):
    path = output_dir / f"final_eval_{timestamp}.csv"
    fieldnames = [
        "model", "bot_level", "n_agents", "num_games",
        "trained_wins", "win_rate_pct", "avg_vp_trained", "avg_vp_bot",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model": r["model"],
                "bot_level": r["bot_level"],
                "n_agents": r["num_trained"],
                "num_games": r["num_games"],
                "trained_wins": r["trained_wins"],
                "win_rate_pct": round(r["win_rate"] * 100, 2),
                "avg_vp_trained": round(r["avg_vp_trained"], 2),
                "avg_vp_bot": round(r["avg_vp_bot"], 2),
            })
    print(f"CSV   → {path}")


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    model_variants: dict[str, str] = {}
    if args.model in ("base", "both"):
        model_variants["base"] = args.base_model
    if args.model in ("ft", "both"):
        model_variants["ft"] = args.ft_model

    bot_levels = [int(x.strip()) for x in args.bot_levels.split(",")]
    agent_counts = [int(x.strip()) for x in args.agent_counts.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "timestamp": timestamp,
        "num_games_per_config": args.num_games,
        "seed": args.seed,
        "base_model_path": args.base_model,
        "ft_model_path": args.ft_model,
        "bot_levels": bot_levels,
        "agent_counts": agent_counts,
    }

    total_configs = len(model_variants) * len(bot_levels) * len(agent_counts)
    all_results: list[dict] = []

    print(f"\nRunning {total_configs} configurations × {args.num_games} games each\n")

    with tqdm(total=total_configs, desc="Configurations", unit="cfg") as pbar:
        for model_tag, model_path in model_variants.items():
            for bot_level in bot_levels:
                for num_trained in agent_counts:
                    pbar.set_postfix(model=model_tag, level=bot_level, agents=num_trained)
                    result = run_configuration(
                        model_tag=model_tag,
                        model_path=model_path,
                        bot_level=bot_level,
                        num_trained=num_trained,
                        num_games=args.num_games,
                        seed=args.seed,
                    )
                    all_results.append(result)
                    pbar.update(1)

    print("\n" + format_table(all_results) + "\n")
    save_json(all_results, output_dir, timestamp, metadata)
    save_csv(all_results, output_dir, timestamp)
    print(f"\nDone. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
