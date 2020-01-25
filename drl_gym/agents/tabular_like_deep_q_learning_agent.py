from tensorflow.python.keras.metrics import *
from tensorflow.python.keras.utils import *

from drl_gym.brains import DQNBrain
from drl_gym.contracts import Agent, GameState


class TabularLikeDeepQLearningAgent(Agent):
    def __init__(
        self,
        action_space_size: int,
        alpha: float = 0.01,
        gamma: float = 0.999,
        epsilon: float = 0.1,
    ):
        self.Q = DQNBrain(output_dim=action_space_size, learning_rate=alpha)
        self.action_space_size = action_space_size
        self.s = None
        self.a = None
        self.r = None
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, gs: GameState) -> int:
        gs_unique_id = gs.get_unique_id()
        available_actions = gs.get_available_actions(gs.get_active_player())

        state_vec = to_categorical(gs_unique_id, gs.get_max_state_count())
        predicted_Q_values = self.Q.predict(state_vec)

        if np.random.random() <= self.epsilon:
            chosen_action = np.random.choice(available_actions)
        else:
            chosen_action = available_actions[
                int(np.argmax(predicted_Q_values[available_actions]))
            ]

        if self.s is not None:
            target = self.r + self.gamma * max(predicted_Q_values[available_actions])
            self.Q.train(self.s, self.a, target)

        self.s = state_vec
        self.a = to_categorical(chosen_action, self.action_space_size)
        self.r = 0.0

        return chosen_action

    def observe(self, r: float, t: bool, player_index: int):
        if self.r is None:
            return

        self.r += r

        if t:
            target = self.r
            self.Q.train(self.s, self.a, target)
            self.s = None
            self.a = None
            self.r = None

    def save_model(self, name: str):
        pass
