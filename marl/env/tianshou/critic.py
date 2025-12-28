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

    def forward(self, obs):
        obs_vec = obs["obs"]
        x, _ = self.net(obs_vec)
        return self.value(x)
