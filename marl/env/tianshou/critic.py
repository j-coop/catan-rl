import torch
import torch.nn as nn
from tianshou.utils.net.common import Net

from params.catan_constants import FINAL_LATENT


class Critic(nn.Module):

    def __init__(self, obs_dim, encoder=None):
        super().__init__()

        self.encoder = encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = FINAL_LATENT if encoder is not None else obs_dim

        self.net = Net(
            state_shape=input_dim,
            hidden_sizes=[2048, 1024, 512, 512],
            activation=nn.ReLU,
        )
        self.value = nn.Linear(512, 1)
        self.to(self.device)

    def forward(self, obs):
        if hasattr(obs, "obs"):
            obs = obs.obs

        if isinstance(obs, dict):
            obs_vec = obs.get("observation", obs.get("obs"))
        else:
            obs_vec = obs

        if not torch.is_tensor(obs_vec):
            obs_vec = torch.tensor(obs_vec, dtype=torch.float32, device=self.device)
        else:
            obs_vec = obs_vec.to(self.device, dtype=torch.float32)

        if self.encoder is not None:
            with torch.no_grad():
                obs_vec = self.encoder.encode(obs_vec)

        x, _ = self.net(obs_vec)
        return self.value(x)
