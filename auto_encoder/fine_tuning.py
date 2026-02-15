from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from marl.env.tianshou.actor import MaskedActor
from marl.env.tianshou.multi_agent_env import CatanEnv
from auto_encoder.encoders import CatanFactorizedAutoEncoder


# ---------------- CONFIG ----------------
NUM_GAMES = 6000
STATE_SKIP = 6
BATCH_SIZE = 32
LR = 1e-4                # small LR for fine-tuning
EPOCHS = 30
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent

AE_PATH = BASE_DIR / "trained_models" / "catan_autoencoder.pth"
AE_SAVE_PATH = BASE_DIR / "trained_models" / "catan_autoencoder_finetuned.pth"

POLICY_PATH = BASE_DIR / "marl" / "env" / "tianshou" / "trained_models" / "marl_model.pt"


# ---------------- LOAD ACTOR ----------------
def load_actor(env: CatanEnv, model_path: str):
    obs_dim = env.get_observation_space_size()
    act_dim = env.actions.get_action_space_size()

    actor = MaskedActor(obs_dim, act_dim)
    state = torch.load(model_path, map_location=DEVICE)

    if isinstance(state, dict):
        state = state.get("policy", state.get("actor", state))

    if any(k.startswith("actor.") for k in state.keys()):
        state = {k[len("actor."):]: v for k, v in state.items()}

    actor.load_state_dict(state)
    actor.to(DEVICE)
    actor.eval()
    return actor


# ---------------- ACTION SELECTION ----------------
def select_action(agent_name, env, actor):
    player = env.game.get_player(agent_name)
    mask = env.actions.get_action_mask(player)

    obs_vec = env.get_observation(agent_name)
    obs_vec = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    obs = {"observation": obs_vec, "action_mask": mask_tensor}

    with torch.no_grad():
        logits, _ = actor(obs)

    logits[mask_tensor == 0] = -1e9
    action = torch.argmax(logits, dim=1).item()

    return action


# ---------------- COLLECT STATES ----------------
def collect_states():
    print("Collecting states from 4-agent self-play...")
    env = CatanEnv()
    actor = load_actor(env, POLICY_PATH)

    all_states = []

    for _ in tqdm(range(NUM_GAMES)):
        env.reset()
        step_counter = 0

        while not env.game.game_over:
            current = env.agent_selection
            action = select_action(current, env, actor)
            obs = env.get_observation(current)
            if step_counter % STATE_SKIP == 0:
                all_states.append(obs)

            # apply action
            for spec in env.actions.action_specs:
                start, end = spec.range
                if start <= action < end:
                    local_index = action - start
                    if spec.name == "end_turn":
                        env.game.end_turn(is_ui_action=False)
                    else:
                        spec.handler(current, local_index)
                    break

            if spec.name == "end_turn":
                env.agent_selection = env.game.current_player.name
                env.game.handle_dice_roll()

            step_counter += 1

    print(f"Collected {len(all_states)} states.")
    return torch.tensor(np.array(all_states), dtype=torch.float32)


# ---------------- FINE-TUNE AUTOENCODER ----------------
def finetune_autoencoder(states_tensor):
    dataset = TensorDataset(states_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    autoencoder = CatanFactorizedAutoEncoder().to(DEVICE)
    autoencoder.load_state_dict(torch.load(AE_PATH, map_location=DEVICE))

    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    print("Fine-tuning autoencoder...")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(DEVICE)

            recon, z = autoencoder(x)

            recon_loss = F.mse_loss(recon, x)

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            epoch_loss += recon_loss.item() * x.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

    torch.save(autoencoder.state_dict(), AE_SAVE_PATH)
    print(f"Fine-tuned model saved to {AE_SAVE_PATH}")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    states = collect_states()
    finetune_autoencoder(states)
