from marl.model.CatanGame import CatanGame
import numpy as np

from params.catan_constants import DICE_PROBABILITIES
from params.nodes2tiles_adjacency_map import NODES_TO_TILES


def token_probability(token):
    return DICE_PROBABILITIES.get(token, 0)


class Rewards:
    def __init__(self, game: CatanGame):
        self.game = game

        self.resource_bias = {
            "wood": 1.1,
            "brick": 1.1,
            "sheep": 0.8,
            "wheat": 1.1,
            "ore": 0.9
        }

    def compute_potential(self, agent):
        player = self.game.players[agent]

        vp_component = player.victory_points / 10.0
        prod_component = self.expected_production(player)
        safety_component = self.risk_penalty(player)
        diversity_component = self.resource_diversity(player)
        dev_potential = self.dev_card_value(player)
        map_potential = self.map_positional_value(player)

        return (
            1.0 * vp_component +
            0.4 * prod_component +
            0.2 * diversity_component +
            0.3 * dev_potential +
            0.25 * map_potential +
            -0.15 * safety_component
        )

    def expected_production(self, player):
        """
        Computes production potential:
          - Quantity: weighted expected production per resource
          - Entropy: diversity of production (balanced economy)
        """

        prod_by_resource = {r: 0.0 for r in self.resource_bias.keys()}

        # Accumulate production from settlements and cities
        for node in player.settlements:
            for r, p in self.production_at_node(node).items():
                prod_by_resource[r] += p

        for node in player.cities:
            for r, p in self.production_at_node(node).items():
                prod_by_resource[r] += 2.0 * p

        # Quantity with bias
        quantity = sum(
            prod_by_resource[r] * self.resource_bias[r]
            for r in prod_by_resource
        )

        # Normalize for maximum expected possible production
        quantity_norm = quantity / 15.0  # TO BE SET

        # Entropy (production diversity)
        total = sum(prod_by_resource.values())
        if total > 0:
            probs = np.array([v / total for v in prod_by_resource.values()])
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            entropy /= np.log(5)  # normalize to [0,1]
        else:
            entropy = 0.0

        return quantity_norm + 0.4 * entropy

    def production_at_node(self, node_index):
        """
        Returns dict: resource -> expected production value for this node.
        """
        result = {r: 0.0 for r in self.resource_bias.keys()}

        tile_indices = NODES_TO_TILES[node_index]
        for tid in tile_indices:
            (res, token) = self.game.board.tiles[tid]
            if token is None or res == "desert":
                continue
            prob = token_probability(token)
            result[res] += prob

        return result

    def risk_penalty(self, player):
        pass

    def resource_diversity(self, player):
        pass

    def dev_card_value(self, player):
        pass

    def map_positional_value(self, player):
        pass

    
