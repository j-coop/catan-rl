import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from marl.env.ActionSpace import ActionSpace
from marl.env.Rewards import Rewards
from marl.env.common import EnvActionHandlerMixin
from params.catan_constants import *
from marl.model.CatanGame import CatanGame


class CatanEnv(MultiAgentEnv,
               EnvActionHandlerMixin):

    metadata = {"name": "catan_v0"}

    def __init__(self, env_config=None):
        super().__init__()
        self.colors = [""] * 4
        self.agents = [
            "Blue Player",
            "Purple Player",
            "Yellow Player",
            "Green Player"
        ]
        self.agent_selection = self.agents[0]
        self.possible_agents = self.agents.copy()

        # Game Logic Layer object
        self.game = CatanGame(player_colors=self.colors,
                              player_names=self.agents)

        self.actions = ActionSpace(self)

        self.observation_spaces = {
            agent: spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.get_observation_space_size(),),
                        dtype=np.float32
                    )
            for agent in self.agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(self.actions.get_action_space_size())
            for agent in self.agents
        }

        self.reward_object = Rewards(self.game)

        # First roll is handled manually
        self.pending_dice_roll = False
        self.game.handle_dice_roll()

    @property
    def get_sub_environments(self):
        return self.unwrapped

    def seed(self, seed=None):
        """Set the RNG seed for reproducibility."""
        np.random.seed(seed)
        return [seed]

    @staticmethod
    def get_observation_space_size() -> int:
        """Return the flattened size of the observation vector."""
        return FULL_ACTION_SPACE_SIZE

    """
    Handles executing action with given index in action space for given agent
    Calls delegating logic to game object logic layer (CatanGame)
    """
    def apply_action(self, agent: str, action: int):
        for spec in self.actions.action_specs:
            start, end = spec.range
            if start <= action < end:
                local_index = action - start
                spec.handler(agent, local_index)
                return
        raise ValueError(f"Invalid action index: {action}")

    """
    One step corresponds to one action (finer control, better action to reward association)
    Only choosing 'end turn' action ends game logic turn
    """
    def step(self, action_dict):

        agent = self.agent_selection
        player = self.game.get_player(agent)
        action = action_dict[agent]
        if VERBOSE:
            print("------------------------------------------------------------------------------------")
            print(f"TURN {self.game.turn} - {self.agent_selection}")
            print(f"Resources: {player.resources}")
            print(f"Chosen action: {action}")

        # minimum na teraz - wybiera akcje, nielegalne kończą turę - ponoć nawet stosowane
        mask = self.actions.get_action_mask(player)
        if mask[action] == 0:
            illegal_penalty = -5
            self.rewards[agent] = illegal_penalty
            self._cumulative_rewards[agent] += illegal_penalty

            # Observation does NOT advance game
            obs = {agent: self.observe(agent)}

            rewards = {p.name: 0.0 for p in self.game.players}
            rewards[agent] = illegal_penalty

            terminateds = {p.name: False for p in self.game.players}
            terminateds["__all__"] = False

            truncateds = {p.name: False for p in self.game.players}
            truncateds["__all__"] = False

            infos = {
                agent: {
                    "illegal_action": True,
                    "attempted_action": int(action),
                }
            }
            return obs, rewards, terminateds, truncateds, infos

        potential_before = self.compute_potential(agent)
        print(f"Resources before apply_action: {player.resources}")
        self.apply_action(agent, action)

        # Check if this ends the current player's turn
        if self.is_end_turn_action(action):
            # Advance to next player (no dice roll yet)
            self.game.end_turn(is_ui_action=False)
            self.agent_selection = self.game.current_player.name
            self.pending_dice_roll = True
        else:
            # Continue with same agent
            self.agent_selection = agent

        potential_after = self.compute_potential(agent)
        reward = self.compute_reward(agent, potential_before, potential_after)
        self.rewards[agent] = reward

        # IMPORTANT for PettingZoo bookkeeping:
        # self._accumulate_rewards()
        self._cumulative_rewards[agent] += self.rewards[agent]

        # Handle dice roll if necessary
        if self.pending_dice_roll:
            self.game.handle_dice_roll()
            self.pending_dice_roll = False

        # Observations
        obs = {self.agent_selection: self.observe(self.agent_selection)}

        # RLlib expects a dict of rewards for all agents
        rewards = {p.name: self.rewards.get(p.name, 0.0) for p in self.game.players}

        # RLlib expects terminateds, truncateds dict with "__all__"
        terminateds = {p.name: self.terminations.get(p.name, False) for p in self.game.players}
        terminateds["__all__"] = all(terminateds.values())

        truncateds = {p.name: self.truncations.get(p.name, False) for p in self.game.players}
        truncateds["__all__"] = all(truncateds.values())

        # Info dict per agent
        infos = {self.agent_selection: self.infos.get(self.agent_selection, {})}

        return obs, rewards, terminateds, truncateds, infos

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.game = CatanGame(player_colors=self.colors,
                              player_names=self.agents)

        self.actions = ActionSpace(self)

        self.agent_selection = self.agents[0]
        self._agent_iterator = iter(self.agents)

        # REQUIRED for PettingZoo
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Handle initial dice roll
        # We want the first player to begin after a dice roll occurs.
        self.game.handle_dice_roll()
        obs = {self.agent_selection: self.observe(self.agent_selection)}
        return obs, {}

    def observe(self, agent):
        """
        Return the observation dict for `agent`.
        We return:
            {
                "observation": np.array([...], dtype=np.float32),
                "action_mask": np.array([...], dtype=np.int8)
            }
        This function expects CatanGame to provide:
         - game.get_observation(agent) -> numpy array
        """
        obs_vec = np.array(self.get_observation(agent), dtype=np.float32)
        return obs_vec

    def render(self):
        pass

    def state(self):
        pass

    def close(self):
        pass
