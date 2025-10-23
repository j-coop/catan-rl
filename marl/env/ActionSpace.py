from marl.model.CatanGame import CatanGame
from marl.model.CatanPhase import CatanPhase
from marl.model.CatanPlayer import CatanPlayer
from marl.util.ActionSpec import ActionSpec
from params.catan_constants import N_NODES, N_EDGES


class ActionSpace:
    def __init__(self, game: CatanGame):

        self.game = game

        # Mapping of action space to action handlers
        self.action_specs: list[ActionSpec] = []
        # Action specs initialization
        self.action_space_size = 0
        self.init_action_specs()

    @staticmethod
    def get_action_space_size() -> int:
        size = 0
        # N_NODES for placing settlements and cities each
        size += 2 * N_NODES
        # N_EDGES for placing roads
        size += N_EDGES
        # 1 for buy dev card
        # 5 for playing dev cards
        # 19 for moving robber to each field (steal included)
        # 20 for trading with bank (each resource for each resource)
        # 5 for choosing resource (year of plenty, monopoly)
        # 1 for end turn
        size += 1 + 5 + 19 + 20 + 1
        return size

    def init_action_specs(self):
        start = 0

        self.action_specs.append(ActionSpec("build_settlement", (start, start + N_NODES), self.game.build_settlement))
        start += N_NODES

        self.action_specs.append(ActionSpec("build_city", (start, start + N_NODES), self.game.build_city))
        start += N_NODES

        self.action_specs.append(ActionSpec("build_road", (start, start + N_EDGES), self.game.build_road))
        start += N_EDGES

        self.action_specs.append(ActionSpec("buy_dev_card", (start, start + 1), self.game.buy_dev_card))
        start += 1

        self.action_specs.append(ActionSpec("play_dev_card", (start, start + 5), self.game.play_dev_card))
        start += 5

        self.action_specs.append(ActionSpec("move_robber", (start, start + 19), self.game.move_robber))
        start += 19

        self.action_specs.append(ActionSpec("trade_bank", (start, start + 20), self.game.trade_bank))
        start += 20

        self.action_specs.append(ActionSpec("choose_resource", (start, start + 5), self.game.choose_resource))
        start += 5

        self.action_specs.append(ActionSpec("end_turn", (start, start + 1), self.game.end_turn))
        start += 1

        self.action_space_size = start

    # Utility to enable actions dynamically
    def _enable(self, mask: list[bool], name: str):
        spec = next((s for s in self.action_specs if s.name == name), None)
        if spec:
            start, end = spec.range
            mask[start:end] = [True] * (end - start)

    def get_action_mask(self) -> list[bool]:
        """
        Returns a boolean mask over the action space,
        enabling only valid actions for the current phase.
        """
        mask = [False] * self.action_space_size
        phase = self.game.phase
        player = self.game.current_player

        # Phase-based logic
        if phase == CatanPhase.NORMAL:
            self._apply_normal_phase_mask(mask, player)
        elif phase == CatanPhase.ROBBER_MOVE:
            self._enable(mask, "move_robber")
        elif phase == CatanPhase.YEAR_OF_PLENTY:
            self._enable(mask, "choose_resource")
        elif phase == CatanPhase.MONOPOLY:
            self._enable(mask, "choose_resource")
        elif phase == CatanPhase.ROAD_BUILDING:
            self._enable(mask, "build_road")

        return mask

    def _apply_normal_phase_mask(self, mask: list[bool], player: CatanPlayer):
        """
        Enable actions based on affordability and legality
        """

        # --- Building settlements ---
        if player.can_afford("settlement"):
            valid_nodes = self.game.board.get_valid_settlement_spots(player)
            spec = next(s for s in self.action_specs if s.name == "build_settlement")
            for node in valid_nodes:
                mask[spec.range[0] + node] = True

        # --- Building cities ---
        if player.can_afford("city"):
            # player can only upgrade existing settlements
            spec = next(s for s in self.action_specs if s.name == "build_city")
            for node in player.settlements:
                mask[spec.range[0] + node] = True

        # --- Building roads ---
        if player.can_afford("road"):
            valid_edges = self.game.board.get_valid_road_spots(player)
            spec = next(s for s in self.action_specs if s.name == "build_road")
            for edge in valid_edges:
                mask[spec.range[0] + edge] = True

        # --- Buying dev card ---
        if player.can_afford("dev_card") and not self.game.bank.remaining_dev_cards() > 0:
            self._enable(mask, "buy_dev_card")

        # --- Playing dev cards ---
        playable_cards = player.get_playable_dev_cards()
        spec = next(s for s in self.action_specs if s.name == "play_dev_card")
        for i, card_type in enumerate(["knight", "victory_point", "road_building", "year_of_plenty", "monopoly"]):
            if card_type in playable_cards:
                mask[spec.range[0] + i] = True

        # --- Trading with bank ---
        trade_pairs = self._get_valid_bank_trades(player)
        spec = next(s for s in self.action_specs if s.name == "trade_bank")
        for idx in trade_pairs:
            mask[spec.range[0] + idx] = True

        # --- Always allow end turn ---
        self._enable(mask, "end_turn")

