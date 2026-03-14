import torch
import torch.nn as nn
import torch.nn.functional as F

from params.catan_constants import (BOARD_SPACE_SIZE,
                                    OTHERS_SPACE_SIZE,
                                    SELF_SPACE_SIZE)


class BoardEncoder(nn.Module):
    def __init__(self, input_dim=BOARD_SPACE_SIZE, latent_dim=512):
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
    def __init__(self, input_dim=SELF_SPACE_SIZE, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class OthersEncoder(nn.Module):
    def __init__(self, input_dim=OTHERS_SPACE_SIZE, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class FusionEncoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),

            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.net(x)


class CatanFactorizedAutoEncoder(nn.Module):
    def __init__(self,
                 board_latent=512,
                 self_latent=128,
                 others_latent=128,
                 final_latent=1024):

        super().__init__()

        self.board_encoder = BoardEncoder(BOARD_SPACE_SIZE, board_latent)
        self.self_encoder = SelfEncoder(SELF_SPACE_SIZE, self_latent)
        self.others_encoder = OthersEncoder(OTHERS_SPACE_SIZE, others_latent)

        self.fusion_encoder = FusionEncoder(
            board_latent + self_latent + others_latent,
            final_latent
        )

        # 🔥 NEW: decode from fusion latent
        self.global_decoder = nn.Sequential(
            nn.Linear(final_latent, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048,
                      BOARD_SPACE_SIZE + SELF_SPACE_SIZE + OTHERS_SPACE_SIZE),
            nn.Sigmoid()
        )

    def split_input(self, x):
        board = x[:, :BOARD_SPACE_SIZE]
        self_info = x[:, BOARD_SPACE_SIZE:(BOARD_SPACE_SIZE + SELF_SPACE_SIZE)]
        others = x[:, (BOARD_SPACE_SIZE + SELF_SPACE_SIZE):]
        return board, self_info, others

    def forward(self, x):
        board, self_info, others = self.split_input(x)

        board_embd = self.board_encoder(board)
        self_embd = self.self_encoder(self_info)
        others_embd = self.others_encoder(others)

        fused = torch.cat([board_embd, self_embd, others_embd], dim=1)
        z = self.fusion_encoder(fused)
        recon = self.global_decoder(z)
        return recon, z

    def encode(self, x):
        board, self_info, others = self.split_input(x)

        board_lat = self.board_encoder(board)
        self_lat = self.self_encoder(self_info)
        others_lat = self.others_encoder(others)

        fused = torch.cat([board_lat, self_lat, others_lat], dim=1)
        return self.fusion_encoder(fused)
