import torch
import torch.nn as nn
from tianshou.utils.net.common import Net


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = Net(
            state_shape=obs_dim,
            hidden_sizes=[512, 512],
            activation=nn.ReLU,
        )
        self.value = nn.Linear(512, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        x, _ = self.net(obs_vec)
        return self.value(x)
