import torch
from tianshou.utils.net.common import Net

from params.catan_constants import FINAL_LATENT


class MaskedActor(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, encoder=None):
        super().__init__()

        self.encoder = encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = FINAL_LATENT if encoder is not None else obs_dim

        self.net = Net(
            state_shape=input_dim,
            hidden_sizes=[2048, 1024, 512, 512],
            activation=torch.nn.ReLU,
        )
        self.logits = torch.nn.Linear(512, action_dim)
        self.to(self.device)

    def forward(self, obs, state=None, info=None):
        if hasattr(obs, "obs"):
            obs_vec = obs.obs
            mask = obs.mask
        else:
            obs_vec = obs["observation"]
            mask = obs["action_mask"]

        if not torch.is_tensor(obs_vec):
            obs_vec = torch.tensor(obs_vec, dtype=torch.float32, device=self.device)
        else:
            obs_vec = obs_vec.to(self.device, dtype=torch.float32)

        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(self.device, dtype=torch.bool)

        if self.encoder is not None:
            with torch.no_grad():
                obs_vec = self.encoder.encode(obs_vec)

        x, _ = self.net(obs_vec, state)
        logits = self.logits(x)

        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, state
