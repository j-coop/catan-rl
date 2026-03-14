import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import itertools

from marl.env.tianshou.multi_agent_env import CatanEnv
from auto_encoder.encoders import CatanFactorizedAutoEncoder

from params.catan_constants import (
    BOARD_LATENT,
    FINAL_LATENT,
    OTHERS_LATENT,
    SELF_LATENT
)

# ---------------- CONFIG ----------------
NUM_GAMES = 10000
STATE_SKIP = 12
BATCH_SIZE = 64
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPERATURES = [0.05, 0.07, 0.1]
LRS = [3e-4, 1e-4]

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "trained_models"
MODEL_DIR.mkdir(exist_ok=True)


# ---------------- DATASET ----------------
class ContrastiveCatanDataset(Dataset):
    def __init__(self, all_games, state_skip=STATE_SKIP):
        self.games = [list(game) for game in all_games]
        self.state_skip = state_skip
        self.total_states = sum(len(game) for game in self.games)

    def __len__(self):
        return self.total_states

    def __getitem__(self, idx):
        valid_games = [g for g in self.games if len(g) > 1]
        anchor_game = random.choice(valid_games)

        anchor_idx = random.randint(0, len(anchor_game)-1)
        anchor = np.array(anchor_game[anchor_idx], dtype=np.float32)

        # positive
        start = max(0, anchor_idx - self.state_skip)
        end = min(len(anchor_game)-1, anchor_idx + self.state_skip)
        pos_candidates = [i for i in range(start, end+1) if i != anchor_idx]
        pos_idx = random.choice(pos_candidates)
        positive = np.array(anchor_game[pos_idx], dtype=np.float32)

        # negative (different game)
        other_games = [g for g in self.games if g is not anchor_game and len(g) > 0]
        neg_game = random.choice(other_games)
        negative_idx = random.randint(0, len(neg_game)-1)
        negative = np.array(neg_game[negative_idx], dtype=np.float32)

        return (
            torch.tensor(anchor),
            torch.tensor(positive),
            torch.tensor(negative)
        )


# ---------------- COLLECT STATES ----------------
def collect_all_games(num_games=NUM_GAMES):
    env = CatanEnv()
    all_games = []

    print("Collecting states once...")
    for _ in tqdm(range(num_games)):
        env.reset()
        done = False
        step_counter = 0
        game_states = []

        while not done:
            agent = env.agent_selection
            obs = env.get_observation(agent)

            if step_counter % STATE_SKIP == 0:
                game_states.append(obs)

            legal_mask = np.array(
                env.actions.get_action_mask(env.game.get_player(agent))
            )
            legal_actions = np.where(legal_mask > 0)[0]
            action = 0 if len(legal_actions) == 0 else np.random.choice(legal_actions)

            env.step(action)
            step_counter += 1
            done = env.game.game_over

        all_games.append(game_states)

    return all_games


# ---------------- INFO-NCE LOSS ----------------
def info_nce_loss(anchor, positive, negative, temperature):
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negative = F.normalize(negative, dim=1)

    pos_sim = torch.sum(anchor * positive, dim=1) / temperature
    neg_sim = torch.sum(anchor * negative, dim=1) / temperature

    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

    return F.cross_entropy(logits, labels)


# ---------------- TRAIN ONE MODEL ----------------
def train_one_model(dataset, lr, temperature):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CatanFactorizedAutoEncoder(
        board_latent=BOARD_LATENT,
        self_latent=SELF_LATENT,
        others_latent=OTHERS_LATENT,
        final_latent=FINAL_LATENT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"\nTraining model | LR={lr} | TEMP={temperature}")

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for anchor, positive, negative in tqdm(dataloader, leave=False):
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            _, z_anchor = model(anchor)
            _, z_positive = model(positive)
            _, z_negative = model(negative)

            loss = info_nce_loss(z_anchor, z_positive, z_negative, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * anchor.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

    return model


# ---------------- MAIN ----------------
if __name__ == "__main__":

    # 1️⃣ Collect once
    all_games = collect_all_games(NUM_GAMES)
    dataset = ContrastiveCatanDataset(all_games)

    # 2️⃣ Train 6 models
    for lr, temp in itertools.product(LRS, TEMPERATURES):

        model = train_one_model(dataset, lr, temp)

        save_path = MODEL_DIR / f"catan_contrastive_lr{lr}_temp{temp}.pth"
        torch.save(model.state_dict(), save_path)

        print(f"Saved model -> {save_path}")
