import math

from marl.model.CatanGame import CatanGame
import numpy as np

from params.catan_constants import DICE_PROBABILITIES
from params.nodes2tiles_adjacency_map import NODES_TO_TILES
from params.tiles2nodes_adjacency_map import TILES_TO_NODES


def token_probability(token):
    return DICE_PROBABILITIES.get(token, 0)


class Rewards:
    def __init__(self, game: CatanGame):
        self.game = game

        # Slight bias towards resources needed for initial growth
        self.resource_bias = {
            "wood": 1.1,
            "brick": 1.1,
            "sheep": 0.8,
            "wheat": 1.1,
            "ore": 0.9
        }

    def compute_potential(self, agent):
        player = self.game.players[agent]

        vp_component = player.victory_points / 10.0 # strongest signal
        prod_component = self.expected_production(player) # production quantity and entropy
        resource_component = self.resource_component(player) # current resources leverage
        risk_component = self.risk_penalty(player) #
        dev_potential = self.dev_card_value(player)
        map_potential = self.map_positional_value(player)

        return (
            1.0 * vp_component +
            0.4 * prod_component +
            0.2 * resource_component +
            0.3 * dev_potential +
            0.25 * map_potential +
            -0.15 * risk_component
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
        quantity_norm = max(quantity / 15.0, 1.0)  # TODO: TO BE SET

        # Entropy (production diversity)
        total = sum(prod_by_resource.values())
        if total > 0:
            probs = np.array([v / total for v in prod_by_resource.values()])
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            entropy /= np.log(5)  # normalize to [0,1]
        else:
            entropy = 0.0

        return 0.6 * quantity_norm + 0.4 * entropy

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

    @staticmethod
    def resource_component(player):
        """
        Returns points value as total for player's resources
        Not normalized not to underestimate very strong hands
        Gives stronger marks for initial resources (encouraging diversity needed to build)
        0.3 points for first resource of type, 0.2 for second, 0.1 for next ones
        """
        resources = player.resources
        total_strength = 0.0

        for resource_type, count in resources.items():
            if count >= 1:
                total_strength += 0.3  # First resource of type
            if count >= 2:
                total_strength += 0.2  # Second resource of type
            if count >= 3:
                total_strength += 0.1 * (count - 2)  # Third and beyond

        return total_strength

    def risk_penalty(self, player):
        # Risk from losing cards on 7
        card_number_risk = 0
        total_cards = sum(player.resources.values())
        if total_cards >= 7:
            card_number_risk = min((total_cards - 6) * 0.2, 1.0)

        # Penalty from having a tile blocked by knight
        blocked_tile_penalty = 0
        knight_tile_index = self.game.board.robber_position
        node_indices = TILES_TO_NODES[knight_tile_index]
        for node in node_indices:
            if node in player.settlements or node in player.cities:
                tile_token = self.game.board.tiles[node][1]
                if tile_token is not None and tile_token != 7:
                    blocked_tile_penalty = 1.0 / int(math.fabs(7 - tile_token))

        return 0.6 * card_number_risk + 0.4 * blocked_tile_penalty

    def resource_diversity(self, player):
        pass

    def dev_card_value(self, player):
        pass

    def map_positional_value(self, player):
        pass
