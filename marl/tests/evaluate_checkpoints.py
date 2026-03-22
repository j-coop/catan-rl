import argparse
import os
import re
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from marl.env.tianshou.actor import MaskedActor
from marl.env.tianshou.multi_agent_env import CatanEnv

# Reuse logic from agent_vs_random.py
def load_actor(env: CatanEnv, model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if hasattr(env, "get_observation_space_size"):
        obs_dim = env.get_observation_space_size()
    else:
        obs_dim = len(env.get_observation(env.agents[0]))
    act_dim = env.actions.get_action_space_size()
    actor = MaskedActor(obs_dim, act_dim)
    state = torch.load(model_path, map_location=actor.device)
    if isinstance(state, dict):
        state = state.get("policy", state.get("actor", state))
    if isinstance(state, dict) and any(k.startswith("actor.") for k in state.keys()):
        state = {k[len("actor."):]: v for k, v in state.items()}
    actor.load_state_dict(state)
    actor.eval()
    return actor


def select_action(agent_name: str, env: CatanEnv, actor):
    player = env.game.get_player(agent_name)
    mask = env.actions.get_action_mask(player)
    valid_indices = [i for i, v in enumerate(mask) if v]
    if not valid_indices:
        raise RuntimeError(f"No valid actions for {agent_name}")
    if actor is None:
        return random.choice(valid_indices)
    obs_vec = env.get_observation(agent_name)
    if hasattr(obs_vec, "ndim") and obs_vec.ndim == 1:
        obs_vec = obs_vec[None, :]
    if hasattr(mask, "ndim") and mask.ndim == 1:
        mask = [mask]
    obs = {"observation": obs_vec, "action_mask": mask}
    with torch.no_grad():
        logits, _ = actor(obs)
    return int(torch.argmax(logits).item())


def apply_action(agent_name: str, action: int, env: CatanEnv):
    for spec in env.actions.action_specs:
        start, end = spec.range
        if start <= action < end:
            local_index = action - start
            if spec.name == "end_turn":
                env.game.end_turn(is_ui_action=False)
            else:
                spec.handler(agent_name, local_index)
            return spec.name
    raise ValueError(f"Invalid action index: {action}")


def run_games(num_games: int, agent_names: list[str], model_path: str, seed: int | None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    env = CatanEnv()
    actor = load_actor(env, model_path)
    
    wins = {agent: 0 for agent in ["Blue Player", "Purple Player", "Yellow Player", "Green Player"]}
    
    for _ in range(num_games):
        env.reset()
        env.step_counter = 0
        while not env.game.game_over:
            current = env.agent_selection
            if current in agent_names:
                action = select_action(current, env, actor)
            else:
                action = select_action(current, env, None)
            action_type = apply_action(current, action, env)
            if action_type == "end_turn":
                env.agent_selection = env.game.current_player.name
                env.game.handle_dice_roll()
        wins[env.game.winner] = wins.get(env.game.winner, 0) + 1
    return wins


def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints and graph win percentage.")
    parser.add_argument("-n", "--num-games", type=int, default=400, help="Number of games per checkpoint")
    parser.add_argument("--checkpoints-dir", type=str, 
                        default="/home/student/Dokumenty/s184725/magisterka/catan-rl/trained_models\checkpoints",
                        help="Directory containing .pt checkpoints")
    parser.add_argument("--agent-name", type=str, default="Blue Player", help="Name of the player controlled by the agent")
    parser.add_argument("--output-plot", type=str, default="win_rate_evolution.png", help="Path to save the result plot")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max-checkpoints", type=int, default=None, help="Limit number of checkpoints to evaluate for speed")
    args = parser.parse_args()

    checkpoints_path = os.path.abspath(args.checkpoints_dir)
    if not os.path.exists(checkpoints_path):
        # Try relative to repo root if not found
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        checkpoints_path = os.path.join(repo_root, args.checkpoints_dir)
        if not os.path.exists(checkpoints_path):
            raise FileNotFoundError(f"Checkpoints directory not found: {args.checkpoints_dir}")

    # List all .pt files
    files = [f for f in os.listdir(checkpoints_path) if f.endswith(".pt")]
    
    # Sort files numerically
    def extract_number(filename):
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else 0

    files.sort(key=extract_number)

    if args.max_checkpoints:
        files = files[:args.max_checkpoints]

    print(f"Found {len(files)} checkpoints. Evaluating each for {args.num_games} games...")

    checkpoint_nums = []
    win_rates = []

    for filename in tqdm(files, desc="Evaluating checkpoints"):
        full_path = os.path.join(checkpoints_path, filename)
        num = extract_number(filename)
        
        wins = run_games(args.num_games, [args.agent_name], full_path, args.seed)
        total = sum(wins.values())
        agent_wins = wins.get(args.agent_name, 0)
        win_rate = (agent_wins / total) * 100 if total > 0 else 0.0
        
        checkpoint_nums.append(num)
        win_rates.append(win_rate)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(checkpoint_nums, win_rates, marker='o', linestyle='-', color='b')
    plt.title(f"Agent Win Percentage Evolution ({args.agent_name} vs Random)")
    plt.xlabel("Checkpoint Number")
    plt.ylabel("Win Percentage (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 100)
    
    # Add a horizontal line for random chance (25% in 4-player game)
    plt.axhline(y=25, color='r', linestyle='--', label='Random Chance (25%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"Saved plot to {args.output_plot}")
    
    # Also print final summary
    print("\nEvaluation Summary:")
    print(f"{'Checkpoint':<12} | {'Win Rate (%)':<12}")
    print("-" * 27)
    for num, rate in zip(checkpoint_nums, win_rates):
        print(f"{num:<12} | {rate:<12.1f}")

if __name__ == "__main__":
    main()
