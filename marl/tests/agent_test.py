import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
from tianshou.data import Batch

from marl.env.tianshou.actor import MaskedActor
from marl.env.tianshou.multi_agent_env import CatanEnv
from marl.env.tianshou.heuristic_bot import HeuristicCatanPolicy
from auto_encoder.encoders import CatanFactorizedAutoEncoder
from params.catan_constants import (BOARD_LATENT,
                                    FINAL_LATENT,
                                    IS_ENCODER_ENABLED,
                                    OTHERS_LATENT,
                                    SELF_LATENT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "trained_models" / "checkpoints" / "ppo_catan_60.pt"
DEFAULT_ENCODER_PATH = BASE_DIR / "marl" / "env" / "tianshou" / "trained_models" / "catan_contrastive_lr0.0001_temp0.1.pth"

def load_actor(env: CatanEnv, model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    obs_dim = env.get_observation_space_size()
    act_dim = env.actions.get_action_space_size()
    autoencoder = None
    if IS_ENCODER_ENABLED:
        autoencoder = CatanFactorizedAutoEncoder(
            board_latent=BOARD_LATENT,
            self_latent=SELF_LATENT,
            others_latent=OTHERS_LATENT,
            final_latent=FINAL_LATENT
        ).to(DEVICE)
        autoencoder.load_state_dict(torch.load(DEFAULT_ENCODER_PATH, map_location=DEVICE))
    actor = MaskedActor(obs_dim, act_dim, encoder=autoencoder).to(DEVICE)
    state = torch.load(model_path, map_location=actor.device)
    if isinstance(state, dict):
        state = state.get("policy", state.get("actor", state))
    if isinstance(state, dict) and any(k.startswith("actor.") for k in state.keys()):
        state = {k[len("actor."):]: v for k, v in state.items()}
    actor.load_state_dict(state)
    actor.eval()
    return actor

def select_action(agent_name: str, env: CatanEnv, actor, bot_policy):
    player = env.game.get_player(agent_name)
    mask = env.actions.get_action_mask(player)
    
    # If it's the learning agent
    if actor is not None:
        obs_vec = env.get_observation(agent_name)
        if hasattr(obs_vec, "ndim") and obs_vec.ndim == 1:
            obs_vec = obs_vec[None, :]
        obs = {"observation": obs_vec, "action_mask": [mask]}
        with torch.no_grad():
            logits, _ = actor(obs)
        return int(torch.argmax(logits).item())
    
    # If it's the bot
    obs_vec = env.get_observation(agent_name)
    mask_arr = np.array(mask, dtype=int)
    batch = Batch(
        obs=Batch(
            observation=np.array([obs_vec]),
            action_mask=np.array([mask_arr])
        )
    )
    result = bot_policy.forward(batch)
    return int(result.act[0])

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

def run_games(num_games: int, agent_names: list[str], model_path: str, bot_level: int, seed: int | None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    wins = {agent: 0 for agent in ["Blue Player", "Purple Player", "Yellow Player", "Green Player"]}
    env = CatanEnv()
    actor = load_actor(env, model_path)
    bot_policy = HeuristicCatanPolicy(level=bot_level)
    
    for i in range(num_games):
        env.reset()
        while not env.game.game_over:
            current = env.agent_selection
            if current in agent_names:
                action = select_action(current, env, actor, None)
            else:
                action = select_action(current, env, None, bot_policy)
            
            action_type = apply_action(current, action, env)
            if action_type == "end_turn":
                env.agent_selection = env.game.current_player.name
                env.game.handle_dice_roll()
        
        winner = env.game.winner
        wins[winner] += 1
        if (i + 1) % 10 == 0:
            print(f"Game {i+1}/{num_games} finished. Winner: {winner}")
            
    return wins

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-games", type=int, default=100)
    parser.add_argument("--agent-name", type=str, default="Blue Player")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1, help="Heuristic bot level for opponents")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"Testing agent '{args.agent_name}' against Level {args.level} bots...")
    wins = run_games(args.num_games, [args.agent_name], args.model_path, args.level, args.seed)
    
    total = sum(wins.values())
    agent_wins = wins.get(args.agent_name, 0)
    win_rate = agent_wins / total if total else 0.0
    
    print("\n" + "="*30)
    print(f"RESULTS FOR {args.agent_name}")
    print(f"Opponent Level: {args.level}")
    print(f"Total Games: {total}")
    print(f"Agent Wins: {agent_wins}")
    print(f"Agent Win Rate: {win_rate:.3%}")
    print("-" * 30)
    print("Full Scoreboard:", wins)
    print("="*30)

if __name__ == "__main__":
    main()
