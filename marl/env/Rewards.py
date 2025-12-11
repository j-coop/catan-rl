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

        prod_by_resource = {r: 0.0 for r in self.resource_bias.keys()}

        # Accumulate production from settlements and cities
        for node in player.settlements:
            for r, p in self.production_at_node(node).items():
                prod_by_resource[r] += p

        for node in player.cities:
            for r, p in self.production_at_node(node).items():
                prod_by_resource[r] += 2.0 * p

        vp_component = player.victory_points / 10.0 # strongest signal
        prod_component = self.expected_production(prod_by_resource) # production quantity and entropy
        resource_component = self.resource_component(player) # current resources leverage
        risk_component = self.risk_penalty(player) # penalties for too many cards risk, blocked tile
        dev_potential = self.dev_card_value(player) # dev cards potential
        port_potential = self.port_component(player, prod_by_resource) # awards for trade possibilities
        road_component = self.road_component(player) # small road number reward

        vp_weighted = 5.0 * vp_component
        prod_weighted = 1.0 * prod_component
        resource_weighted = 0.3 * resource_component
        dev_weighted = 0.2 * dev_potential
        port_weighted = 0.2 * port_potential
        road_weighted = 0.2 * road_component
        risk_weighted = -0.2 * risk_component

        total_potential = vp_weighted + prod_weighted + resource_weighted + dev_weighted + port_weighted + road_weighted + risk_weighted

        print(
            f"Total: {total_potential:.2f} | "
            f"VP: {vp_weighted:.1f} | "
            f"Prod: {prod_weighted:.1f} | "
            f"Res: {resource_weighted:.1f} | "
            f"Dev: {dev_weighted:.1f} | "
            f"Port: {port_weighted:.1f} | "
            f"Road: {road_weighted:.1f} | "
            f"Risk: {risk_weighted:.1f}"
        )

    def expected_production(self, prod_by_resource):
        """
        Computes production potential:
          - Quantity: weighted expected production per resource
          - Entropy: diversity of production (balanced economy)
        """

        # Quantity with bias
        quantity = sum(
            prod_by_resource[r] * self.resource_bias[r]
            for r in prod_by_resource
        )

        # Normalize for maximum expected possible production
        quantity_norm = max(quantity, 1)

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
        tile_token = self.game.board.tiles[knight_tile_index][1]
        node_indices = TILES_TO_NODES[knight_tile_index]
        if tile_token is not None and tile_token != 7:
            for node in node_indices:
                if node in player.settlements or node in player.cities:
                    blocked_tile_penalty = 1.0 / int(math.fabs(7 - tile_token))

        return 0.6 * card_number_risk + 0.4 * blocked_tile_penalty

    @staticmethod
    def dev_card_value(player):
        """
        Returns points value as total for player's dev cards
        Not normalized not to underestimate very strong hands
        Points also awarded for played knights (if not strongest army already)
        Victory point cards handled in vp_component
        """
        dev = player.dev_cards

        value = (
            0.4 * dev.get("knight", 0) +
            0.3 * player.knights_played +
            0.5 * dev.get("road_building", 0) +
            0.6 * dev.get("monopoly", 0) +
            0.4 * dev.get("year_of_plenty", 0)
        )

        return value

    @staticmethod
    def port_component(player, prod_by_resource):
        """
        Reward for controlled ports
        Value of a port in context of player's resource production
        """
        port_value = 0.0

        for port_type, is_controlled in player.ports.items():
            if is_controlled:
                # If the port type is a specific resource, check if the player produces that resource
                if port_type in prod_by_resource:
                    red_token_equivalent = prod_by_resource[port_type] / DICE_PROBABILITIES.get(6, 5/36)
                    port_value += 0.25 * red_token_equivalent
                else:
                    # Reward for generic 3for1 port
                    port_value += 0.3

        return port_value

    @staticmethod
    def road_component(player):
        """
        Gives small reward for player's total number of roads
        They are useful but usually not rewarded by victory points, cards or resources
        Building road cannot decrease potential or it will be avoided
        Capped at 10 roads
        """
        return max(len(player.roads) * 0.1, 1.0)
