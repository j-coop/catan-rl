import random
import time

from PyQt6.QtWidgets import QApplication

from marl.env.ActionSpace import ActionSpace
from marl.model.CatanGame import CatanGame
from marl.ui.controllers.PlayerController import PlayerController
from params.catan_constants import DEV_CARD_TYPES, BANK_TRADE_PAIRS


class AgentController(PlayerController):
    def __init__(self, player_name, agent, delay=2.0):
        super().__init__(player_name)
        self.player_name = player_name
        self.agent = agent
        self.delay = delay
        self.is_human = False

    def apply_action(self, agent: str, action: int, action_space: ActionSpace, game: CatanGame):
        for spec in action_space.action_specs:
            start, end = spec.range
            if start <= action < end:
                print(f"Action type: {spec.name}")
                local_index = action - start
                print(game.get_player(agent).resources)
                if spec.name == "end_turn":
                    spec.handler(agent, local_index, is_ui_action=True)
                else:
                    spec.handler(agent, local_index)
                return spec, local_index
        raise ValueError(f"Invalid action index: {action}")

    def request_action(self, game, action_space, game_manager):
        # obs = game.get_observation_for_player(self.player_name)
        mask = action_space.get_action_mask(game.get_player(self.player_name))

        # TODO: insert real model action inference here
        valid_indices = [i for i, v in enumerate(mask) if v]
        action = random.choice(valid_indices)

        action_spec, local_index = self.apply_action(self.player_name, action, action_space, game)
        action_type = action_spec.name
        # UI building updates after actions
        color = game.get_player(self.player_name).color
        if action_type == "build_road":
            game_manager.board.build_road_ui(local_index)
            game_manager.log_action(self.player_name, color, f"built road on edge {local_index}")
        elif action_type == "build_settlement":
            game_manager.board.build_settlement_ui(local_index)
            game_manager.log_action(self.player_name, color, f"built settlement on node {local_index}")
        elif action_type == "build_city":
            game_manager.board.upgrade_city_ui(local_index)
            game_manager.log_action(self.player_name, color, f"upgrade to city on node {local_index}")
        elif action_type == "move_robber":
            game_manager.board.update_robber()
            game_manager.log_action(self.player_name, color, f"moved robber to tile {local_index}")
        elif action_type == "buy_dev_card":
            game_manager.log_action(self.player_name, color, f"bought a dev card")
        elif action_type == "play_dev_card":
            game_manager.log_action(self.player_name, color, f"played dev card {DEV_CARD_TYPES[local_index].replace('_', ' ')}")
        elif action_type == "trade_bank":
            give, take = BANK_TRADE_PAIRS[local_index]
            game_manager.log_action(self.player_name, color, f"traded {give} for {take} with bank")
        elif action_type == "end_turn":
            game_manager.log_action(self.player_name, color, f"ended his turn")

        # Update UI state
        game_manager.action_panel.info_panel.refresh()
        game_manager.board.update_roll_display(is_agent=True)
