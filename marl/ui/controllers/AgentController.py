import os
import random

from marl.env.ActionSpace import ActionSpace
from marl.model.CatanGame import CatanGame
from marl.ui.controllers.PlayerController import PlayerController
from params.catan_constants import DEV_CARD_TYPES, BANK_TRADE_PAIRS


class AgentController(PlayerController):
    _actor_model = None
    _actor_ready = False
    _actor_failed = False
    _actor_model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "trained_models", "best_full_game_agent.pt")
    )

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

        self._ensure_actor(action_space)
        action = self._infer_action(action_space, mask)

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

    def _ensure_actor(self, action_space):
        if self.__class__._actor_ready or self.__class__._actor_failed:
            return
        try:
            import torch
            from marl.env.tianshou.actor import MaskedActor
        except Exception:
            self.__class__._actor_failed = True
            print("Actor failed exception")
            return
        if not os.path.exists(self.__class__._actor_model_path):
            print("Path does not exist")
            self.__class__._actor_failed = True
            return
        if hasattr(action_space.env, "get_observation_space_size"):
            obs_dim = action_space.env.get_observation_space_size()
        else:
            obs_dim = len(action_space.env.get_observation(self.player_name))
        act_dim = action_space.get_action_space_size()
        actor = MaskedActor(obs_dim, act_dim)
        state = torch.load(self.__class__._actor_model_path, map_location=actor.device)
        print("Agent model loaded")
        if isinstance(state, dict):
            state = state.get("policy", state.get("actor", state))
        if isinstance(state, dict) and any(k.startswith("actor.") for k in state.keys()):
            state = {k[len("actor."):]: v for k, v in state.items()}
        actor.load_state_dict(state)
        actor.eval()
        self.__class__._actor_model = actor
        self.__class__._actor_ready = True

    def _infer_action(self, action_space, mask):
        valid_indices = [i for i, v in enumerate(mask) if v]
        if not valid_indices or not self.__class__._actor_ready:
            print("Agent not ready - random choice")
            return random.choice(valid_indices)
        obs_vec = action_space.env.get_observation(self.player_name)
        if hasattr(obs_vec, "ndim") and obs_vec.ndim == 1:
            obs_vec = obs_vec[None, :]
        if hasattr(mask, "ndim") and mask.ndim == 1:
            mask = [mask]
        obs = {"observation": obs_vec, "action_mask": mask}
        import torch
        with torch.no_grad():
            logits, _ = self.__class__._actor_model(obs)
        print("Successful inference")
        return int(torch.argmax(logits).item())
