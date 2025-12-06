from marl.model.CatanGame import CatanGame


class Rewards:
    def __init__(self, game: CatanGame):
        self.game = game

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


