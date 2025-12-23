from marl.model.CatanPhase import CatanPhase
from marl.model.CatanPlayer import CatanPlayer
from marl.ui.EnvMock import EnvMock
from marl.util.ActionSpec import ActionSpec
from params.catan_constants import N_NODES, N_EDGES, DEV_CARD_TYPES, BANK_TRADE_PAIRS


class ActionSpace:
    def __init__(self, env):
        self.env = env
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
        size += 1 + 5 + 19 + 20 + 5 + 1
        return size

    def init_action_specs(self):
        start = 0

        # Determine whether to use real callbacks
        use_callbacks = not isinstance(self.env, EnvMock)

        def cb(attr_name):
            if use_callbacks:
                return getattr(self.env, attr_name)
            else:
                return lambda *args, **kwargs: None  # dummy for UI

        self.action_specs.append(ActionSpec("build_settlement", (start, start + N_NODES), cb("build_settlement")))
        start += N_NODES

        self.action_specs.append(ActionSpec("build_city", (start, start + N_NODES), cb("build_city")))
        start += N_NODES

        self.action_specs.append(ActionSpec("build_road", (start, start + N_EDGES), cb("build_road")))
        start += N_EDGES

        self.action_specs.append(ActionSpec("buy_dev_card", (start, start + 1), cb("buy_dev_card")))
        start += 1

        self.action_specs.append(ActionSpec("play_dev_card", (start, start + 5), cb("play_dev_card")))
        start += 5

        self.action_specs.append(ActionSpec("move_robber", (start, start + 19), cb("move_robber")))
        start += 19

        self.action_specs.append(ActionSpec("trade_bank", (start, start + 20), cb("trade_bank")))
        start += 20

        self.action_specs.append(ActionSpec("choose_resource", (start, start + 5), cb("choose_resource")))
        start += 5

        self.action_specs.append(ActionSpec("end_turn", (start, start + 1), cb("end_turn")))
        start += 1
        self.action_space_size = start

    # Utility to enable actions dynamically
    def _enable(self, mask: list[bool], name: str):
        spec = next((s for s in self.action_specs if s.name == name), None)
        if spec:
            start, end = spec.range
            mask[start:end] = [True] * (end - start)

    def get_action_mask(self, player: CatanPlayer) -> list[bool]:
        """
        Returns a boolean mask over the action space,
        enabling only valid actions for the current phase.
        """
        mask = [False] * self.action_space_size
        phase = self.env.game.phase

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
            self._apply_road_building_mask(mask, player)

        return mask

    def _apply_road_building_mask(self, mask: list[bool], player: CatanPlayer):
        """
        Enable valid road placement actions for the Road Building development card.
        Rules are the same as in normal phase, but only roads are allowed.
        """
        valid_edges = self.env.game.board.get_valid_road_spots(player)
        spec = next(s for s in self.action_specs if s.name == "build_road")

        for edge in valid_edges:
            mask[spec.range[0] + edge] = True

    def _apply_normal_phase_mask(self, mask: list[bool], player: CatanPlayer):
        """
        Enable actions based on affordability and legality
        """

        # --- Building settlements ---
        direct = player.can_afford_directly("settlement")
        with_trades = player.can_afford_with_trades("settlement", self.env.game.bank)
        if direct or with_trades:
            valid_nodes = self.env.game.board.get_valid_settlement_spots(player)
            spec = next(s for s in self.action_specs if s.name == "build_settlement")
            for node in valid_nodes:
                mask[spec.range[0] + node] = True

        # --- Building cities ---
        direct = player.can_afford_directly("city")
        with_trades = player.can_afford_with_trades("city", self.env.game.bank)
        if direct or with_trades:
            # player can only upgrade existing settlements
            spec = next(s for s in self.action_specs if s.name == "build_city")
            for node in player.settlements:
                mask[spec.range[0] + node] = True

        # --- Building roads ---
        direct = player.can_afford_directly("road")
        with_trades = player.can_afford_with_trades("road", self.env.game.bank)
        if direct or with_trades:
            valid_edges = self.env.game.board.get_valid_road_spots(player)
            spec = next(s for s in self.action_specs if s.name == "build_road")
            for edge in valid_edges:
                mask[spec.range[0] + edge] = True

        # --- Buying dev card ---
        direct = player.can_afford_directly("dev_card")
        with_trades = player.can_afford_with_trades("dev_card", self.env.game.bank)
        if (direct or with_trades) and self.env.game.bank.remaining_dev_cards() > 0:
            self._enable(mask, "buy_dev_card")

        # --- Playing dev cards ---
        playable_cards = player.get_playable_dev_cards()
        spec = next(s for s in self.action_specs if s.name == "play_dev_card")
        for i, card_type in enumerate(DEV_CARD_TYPES):
            if card_type in playable_cards:
                mask[spec.range[0] + i] = True

        # --- Trading with bank ---
        trade_pairs = player.get_valid_bank_trades()
        spec = next(s for s in self.action_specs if s.name == "trade_bank")
        for trade_pair in trade_pairs:
            idx = BANK_TRADE_PAIRS.index(trade_pair)
            mask[spec.range[0] + idx] = True

        # --- Always allow end turn ---
        self._enable(mask, "end_turn")

    # --- FOR UI ---
    def get_spec(self, name: str) -> ActionSpec:
        return next(s for s in self.action_specs if s.name == name)

    def is_action_enabled(
        self,
        player: CatanPlayer,
        name: str,
        index: int | None = None,
        mask: list[bool] = None,
    ) -> bool:
        """
        If index is None:
            - returns True if ANY action in this range is enabled
        If index is provided:
            - returns True if that specific sub-action is enabled
        """
        if mask is None:
            mask = self.get_action_mask(player)
        spec = self.get_spec(name)
        start, end = spec.range

        if index is None:
            return any(mask[start:end])

        if not (0 <= index < (end - start)):
            raise IndexError(f"Action index {index} out of range for '{name}'")

        return mask[start + index]


