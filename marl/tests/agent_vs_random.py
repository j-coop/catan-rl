import argparse
import os
import random

import torch

from marl.env.tianshou.actor import MaskedActor
from marl.env.tianshou.multi_agent_env import CatanEnv


DEFAULT_MODEL_PATH = os.path.abspath(
    # os.path.join(os.path.dirname(__file__), "..", "..", "trained_models", "best_full_game_agent.pt")
    os.path.join(os.path.dirname(__file__), "..", "env", "tianshou", "trained_models\checkpoints", "ppo_catan_7.pt")
)


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


def run_games(num_games: int, agent_name: str, model_path: str, seed: int | None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    wins = {agent_name: 0}
    env = CatanEnv()
    actor = load_actor(env, model_path)
    for _ in range(num_games):
        env.reset()
        env.step_counter = 0
        while not env.game.game_over:
            current = env.agent_selection
            if current == agent_name:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-games", type=int, default=100)
    parser.add_argument("--agent-name", type=str, default="Blue Player")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    wins = run_games(args.num_games, args.agent_name, args.model_path, args.seed)
    total = sum(wins.values())
    agent_wins = wins.get(args.agent_name, 0)
    win_rate = agent_wins / total if total else 0.0
    print(f"Agent: {args.agent_name}")
    print(f"Games: {total}")
    print(f"Wins: {agent_wins}")
    print(f"Win rate: {win_rate:.3f}")
    print("All wins:", wins)


if __name__ == "__main__":
    main()
