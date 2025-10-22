from marl.model.CatanGame import CatanGame
from marl.model.CatanPhase import CatanPhase
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
            self._enable(mask, "build_settlement")
            self._enable(mask, "build_city")
            self._enable(mask, "build_road")
            self._enable(mask, "buy_dev_card")
            self._enable(mask, "play_dev_card")
            self._enable(mask, "trade_bank")
            self._enable(mask, "end_turn")

        elif phase == CatanPhase.ROBBER_MOVE:
            self._enable(mask, "move_robber")

        elif phase == CatanPhase.YEAR_OF_PLENTY:
            self._enable(mask, "choose_resource")

        elif phase == CatanPhase.MONOPOLY:
            self._enable(mask, "choose_resource")

        elif phase == CatanPhase.ROAD_BUILDING:
            self._enable(mask, "build_road")

        return mask
