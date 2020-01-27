import numpy as np

from tensorflow.python.keras.metrics import *
from tensorflow.python.keras.utils import *

from drl_gym.brains import PPOValueBrain, PPOPolicyBrain
from drl_gym.contracts import Agent, GameState


class PPOAgent(Agent):
    def __init__(
        self,
        state_space_size: int,
        action_space_size: int,
        alpha: float = 0.0001,
        gamma: float = 0.999,
        epsilon: float = 0.1,
        episodes_count_between_training: int = 100,
    ):
        self.critic = PPOValueBrain(
            learning_rate=alpha, hidden_layers_count=5, neurons_per_hidden_layer=128
        )
        self.actor = PPOPolicyBrain(
            learning_rate=alpha,
            state_dim=state_space_size,
            output_dim=action_space_size,
            hidden_layers_count=5,
            neurons_per_hidden_layer=128,
        )
        self.action_space_size = action_space_size
        self.episodes_count_between_training = episodes_count_between_training
        self.s = []
        self.a = []
        self.r = []
        self.v = []
        self.m = []
        self.r_temp = 0.0
        self.is_last_episode_terminal = True
        self.current_episode_count = 0
        self.buffer = {
            "states": [],
            "chosen_actions": [],
            "gains": [],
            "advantages": [],
            "masks": [],
        }
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, gs: GameState) -> int:
        gs_unique_id = gs.get_unique_id()
        available_actions = gs.get_available_actions(gs.get_active_player())

        state_vec = gs.get_vectorized_state()

        mask_vec = np.zeros((self.action_space_size,))
        mask_vec[available_actions] = 1.0

        v = self.critic.predict(state_vec)
        p = self.actor.predict(state_vec, mask_vec)
        p = np.array(p)
        p /= p.sum()
        indexes = np.arange(self.action_space_size)
        chosen_action = np.random.choice(indexes, p=p)

        # valid_actions_probability = p[available_actions]
        # valid_actions_probability_sum = np.sum(valid_actions_probability)
        # normalized_valid_action_probability = valid_actions_probability / valid_actions_probability_sum
        # #
        # chosen_action = np.random.choice(available_actions, p=normalized_valid_action_probability)

        self.v.append(v)

        self.s.append(state_vec)
        self.m.append(mask_vec)
        self.a.append(to_categorical(chosen_action, self.action_space_size))
        if not self.is_last_episode_terminal:
            self.r.append(self.r_temp)
        self.r_temp = 0.0
        self.is_last_episode_terminal = False

        return chosen_action

    def observe(self, r: float, t: bool, player_index: int):
        if self.is_last_episode_terminal:
            return

        self.r_temp += r

        if t:
            self.current_episode_count += 1
            self.r.append(self.r_temp)
            self.compute_gains_and_advantages()
            if self.current_episode_count == self.episodes_count_between_training:
                self.train()
                self.buffer["states"].clear()
                self.buffer["chosen_actions"].clear()
                self.buffer["gains"].clear()
                self.buffer["advantages"].clear()
                self.buffer["masks"].clear()
                self.current_episode_count = 0
            self.s.clear()
            self.a.clear()
            self.r.clear()
            self.v.clear()
            self.m.clear()
            self.r_temp = 0.0
            self.is_last_episode_terminal = True

    def compute_gains_and_advantages(self):

        last_gain = 0.0
        for i in reversed(range(len(self.s))):
            last_gain = self.r[i] + self.gamma * last_gain
            self.buffer["states"].append(self.s[i])
            self.buffer["chosen_actions"].append(self.a[i])
            self.buffer["gains"].append(last_gain)
            self.buffer["advantages"].append(last_gain - self.v[i])
            self.buffer["masks"].append(self.m[i])

    def train(self):
        self.critic.train(
            np.array(self.buffer["states"]), np.array(self.buffer["gains"])
        )
        self.actor.train(
            np.array(self.buffer["states"]),
            np.array(self.buffer["masks"]),
            np.array(self.buffer["chosen_actions"]),
            np.array(self.buffer["advantages"]),
        )

    def save_model(self, filename: str):
        self.actor.save_model(filename)
        self.critic.save_model(filename)

    def load_model(self, filename: str):
        self.actor.load_model(filename)
        self.critic.load_model(filename)
