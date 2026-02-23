import torch
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import (DataLoader,
                              TensorDataset)
import numpy as np
from tqdm import tqdm

from marl.env.tianshou.multi_agent_env import CatanEnv
from auto_encoder.encoders import CatanFactorizedAutoEncoder


NUM_GAMES = 6000
STATE_SKIP = 11
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_SAVE_PATH = BASE_DIR / "trained_models" / "catan_autoencoder.pth"

# ----------------- INIT ENV AND AE -----------------
env = CatanEnv()
autoencoder = CatanFactorizedAutoEncoder().to(DEVICE)

# ----------------- COLLECT STATES -----------------
all_states = []

if '__main__' == __name__:
    print("Collecting states from random games...")
    for game_idx in tqdm(range(NUM_GAMES)):
        env.reset()
        done = False
        step_counter = 0
        
        while not done:
            agent = env.agent_selection
            obs = env.get_observation(agent)
            
            # record state occasionally
            if step_counter % STATE_SKIP == 0:
                all_states.append(obs)
            
            # sample legal action
            legal_mask = np.array(env.actions.get_action_mask(env.game.get_player(agent)))
            legal_actions = np.where(legal_mask > 0)[0]
            action = 0 if len(legal_actions) == 0 else np.random.choice(legal_actions)

            env.step(action)
            step_counter += 1

            # check if game over
            if env.game.game_over:
                done = True

    print(f"Collected {len(all_states)} states.")

    # ----------------- CREATE DATASET -----------------
    states_tensor = torch.tensor(all_states, dtype=torch.float32)
    dataset = TensorDataset(states_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ----------------- TRAIN AUTOENCODER -----------------
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

    print("Pretraining autoencoder (with denoising)...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(DEVICE)

            # ---------- Add noise ----------
            noise_level = 0.04
            x_noisy = x + torch.randn_like(x) * noise_level
            x_noisy = x_noisy.clamp(0.0, 1.0)  # optional: keep features in [0,1]

            recon, z = autoencoder(x_noisy)
            loss = F.mse_loss(recon, x)  # compare to clean original
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
        
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

    # ----------------- SAVE MODEL -----------------
    torch.save(autoencoder.state_dict(), MODEL_SAVE_PATH)
    print(f"Autoencoder saved to {MODEL_SAVE_PATH}")
