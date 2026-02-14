import torch
import torch.nn as nn
import torch.nn.functional as F

from params.catan_constants import (BOARD_SPACE_SIZE,
                                    OTHERS_SPACE_SIZE,
                                    SELF_SPACE_SIZE,
                                    FULL_ACTION_SPACE_SIZE)


class BoardEncoder(nn.Module):
    def __init__(self, input_dim=BOARD_SPACE_SIZE, latent_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class SelfEncoder(nn.Module):
    def __init__(self, input_dim=SELF_SPACE_SIZE, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class OthersEncoder(nn.Module):
    def __init__(self, input_dim=OTHERS_SPACE_SIZE, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class FusionEncoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------- FULL AUTOENCODER ----------

class CatanFactorizedAutoEncoder(nn.Module):
    def __init__(self,
                 board_latent=192,
                 self_latent=32,
                 others_latent=32,
                 final_latent=256):

        super().__init__()

        # Encoders
        self.board_encoder = BoardEncoder(BOARD_SPACE_SIZE, board_latent)
        self.self_encoder = SelfEncoder(SELF_SPACE_SIZE, self_latent)
        self.others_encoder = OthersEncoder(OTHERS_SPACE_SIZE, others_latent)

        self.fusion_encoder = FusionEncoder(
            board_latent + self_latent + others_latent,
            final_latent
        )

        # Decoders (mirror structure)
        self.board_decoder = nn.Sequential(
            nn.Linear(board_latent, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, BOARD_SPACE_SIZE),
            nn.Sigmoid()
        )

        self.self_decoder = nn.Sequential(
            nn.Linear(self_latent, 64),
            nn.ReLU(),
            nn.Linear(64, SELF_SPACE_SIZE),
            nn.Sigmoid()
        )

        self.others_decoder = nn.Sequential(
            nn.Linear(others_latent, 128),
            nn.ReLU(),
            nn.Linear(128, OTHERS_SPACE_SIZE),
            nn.Sigmoid()
        )

    def split_input(self, x):
        board = x[:, :BOARD_SPACE_SIZE]
        self_info = x[:, BOARD_SPACE_SIZE:(BOARD_SPACE_SIZE + SELF_SPACE_SIZE)]
        others = x[:, (BOARD_SPACE_SIZE + SELF_SPACE_SIZE):]
        return board, self_info, others

    def forward(self, x):
        board, self_info, others = self.split_input(x)

        # Encode
        board_embd = self.board_encoder(board)
        self_embd = self.self_encoder(self_info)
        others_embd = self.others_encoder(others)

        fused = torch.cat([board_embd, self_embd, others_embd], dim=1)
        z = self.fusion_encoder(fused)

        # Decode each block independently
        board_recon = self.board_decoder(board_embd)
        self_recon = self.self_decoder(self_embd)
        others_recon = self.others_decoder(others_embd)

        recon = torch.cat([board_recon, self_recon, others_recon], dim=1)
        return recon, z

    def encode(self, x):
        board, self_info, others = self.split_input(x)

        board_lat = self.board_encoder(board)
        self_lat = self.self_encoder(self_info)
        others_lat = self.others_encoder(others)

        fused = torch.cat([board_lat, self_lat, others_lat], dim=1)
        return self.fusion_encoder(fused)
