import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


class CatanPerActionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim=128):
        super().__init__(observation_space, features_dim=hidden_dim)

        # Take one node’s shape from obs dict
        node_dim = 0
        node_dim += int(np.prod(observation_space["tiles_tokens"].shape[1:]))    # (N_ADJ_TILES, N_TOKENS)
        node_dim += int(np.prod(observation_space["tiles_resources"].shape[1:])) # (N_ADJ_TILES, N_RESOURCES)
        node_dim += int(np.prod(observation_space["has_port"].shape[1:]))        # (N_PORT_FIELD_TYPES,)
        node_dim += int(np.prod(observation_space["adj_exist"].shape[1:]))       # (N_ADJ_NODES,)

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
        )

        self._node_dim = node_dim

    def forward(self, obs):
        tiles_tokens = obs["tiles_tokens"].float()
        tiles_resources = obs["tiles_resources"].float()
        has_port = obs["has_port"].float()
        adj_exist = obs["adj_exist"].float()

        B, N = tiles_tokens.shape[0], tiles_tokens.shape[1]

        tokens = tiles_tokens.view(B, N, -1)
        resources = tiles_resources.view(B, N, -1)
        ports = has_port.view(B, N, -1)
        neighbors = adj_exist.view(B, N, -1)

        node_features = th.cat([tokens, resources, ports, neighbors], dim=-1)  # (B, N, F)

        # Pass each node through the MLP
        node_emb = self.node_mlp(node_features)  # (B, N, hidden_dim)

        # Aggregate to fixed-size representation
        graph_emb = node_emb.mean(dim=1)  # (B, hidden_dim)

        return graph_emb

