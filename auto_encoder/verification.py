import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

from marl.env.tianshou.multi_agent_env import CatanEnv
from auto_encoder.encoders import CatanFactorizedAutoEncoder
from params.catan_constants import (BOARD_LATENT,
                                    FINAL_LATENT,
                                    OTHERS_LATENT,
                                    SELF_LATENT)


# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "trained_models" / "catan_contrastive_lr0.0001_temp0.1.pth"

NUM_GAMES = 100
STATE_SKIP = 10

# ---------------- LOAD MODEL ----------------
autoencoder = CatanFactorizedAutoEncoder(
    board_latent=BOARD_LATENT,
    self_latent=SELF_LATENT,
    others_latent=OTHERS_LATENT,
    final_latent=FINAL_LATENT
).to(DEVICE)

autoencoder.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
autoencoder.eval()

print("Model loaded.")

# ---------------- COLLECT TEST STATES ----------------
env = CatanEnv()
states = []

print("Collecting evaluation states...")
for _ in tqdm(range(NUM_GAMES)):
    env.reset()
    done = False
    step_counter = 0

    while not done:
        agent = env.agent_selection
        obs = env.get_observation(agent)

        if step_counter % STATE_SKIP == 0:
            states.append(np.array(obs, dtype=np.float32))

        legal_mask = np.array(
            env.actions.get_action_mask(env.game.get_player(agent))
        )
        legal_actions = np.where(legal_mask > 0)[0]
        action = 0 if len(legal_actions) == 0 else np.random.choice(legal_actions)

        env.step(action)
        step_counter += 1

        if env.game.game_over:
            done = True

states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
print(f"Collected {len(states)} evaluation states.")

# ---------------- 1️⃣ Reconstruction Test ----------------
with torch.no_grad():
    recon, z = autoencoder(states)
    recon_loss = F.mse_loss(recon, states).item()

print(f"\nReconstruction MSE on new data: {recon_loss:.6f}")

# ---------------- 2️⃣ Latent Variance Test ----------------
with torch.no_grad():
    z = autoencoder.encode(states)

latent_std = z.std(dim=0).mean().item()
latent_global_std = z.std().item()

print("\nLatent space statistics:")
print(f"Mean feature std: {latent_std:.6f}")
print(f"Global latent std: {latent_global_std:.6f}")

if latent_std < 1e-3:
    print("WARNING: Possible latent collapse detected.")
else:
    print("Latent variance healthy.")

# ---------------- 3️⃣ Distance Structure Test ----------------
with torch.no_grad():
    z = autoencoder.encode(states)

# pick random pairs
num_samples = 200
indices = torch.randperm(len(z))[:num_samples]

distances_random = []
distances_neighbor = []

for i in range(num_samples - 1):
    # random pair
    a = z[indices[i]]
    b = z[indices[(i + 10) % num_samples]]
    distances_random.append(torch.norm(a - b).item())

    # neighbor pair (adjacent in time)
    a2 = z[i]
    b2 = z[i + 1]
    distances_neighbor.append(torch.norm(a2 - b2).item())

print("\nLatent distance analysis:")
print(f"Average random distance:  {np.mean(distances_random):.4f}")
print(f"Average neighbor distance: {np.mean(distances_neighbor):.4f}")

if np.mean(distances_neighbor) < np.mean(distances_random):
    print("Good: Similar states are closer in latent space.")
else:
    print("Warning: Latent space may not reflect game similarity.")
