import torch
from tianshou.utils.net.common import Net


class MaskedActor(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = Net(
            state_shape=obs_dim,
            hidden_sizes=[512, 512],
            activation=torch.nn.ReLU,
        )
        self.logits = torch.nn.Linear(512, action_dim)

    def forward(self, obs, state=None, info=None):
        obs_vec = obs["obs"]
        mask = obs["mask"]

        if not torch.is_tensor(obs_vec):
            obs_vec = torch.as_tensor(obs_vec, dtype=torch.float32)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, dtype=torch.bool)

        x, _ = self.net(obs_vec, state)
        logits = self.logits(x)

        logits = logits.masked_fill(mask == 0, -1e12)
        return logits, state
