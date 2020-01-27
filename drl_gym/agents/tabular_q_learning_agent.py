import pickle
import numpy as np

from datetime import datetime

from drl_gym.contracts import Agent, GameState


# si gs1 == gs2 => hash(gs1) == hash(gs2)
# si gs1 != gs2 => hash(gs1) != hash(gs2) || hash(gs1) == hash(gs2)


class TabQLearningAgent(Agent):
    def __init__(self, alpha: float = 0.01, gamma: float = 0.999, epsilon: float = 0.1):
        self.Q = dict()
        self.s = None
        self.a = None
        self.r = None
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, gs: GameState) -> int:
        gs_unique_id = gs.get_unique_id()
        available_actions = gs.get_available_actions(gs.get_active_player())
        if gs_unique_id not in self.Q:
            self.Q[gs_unique_id] = dict()
            for a in available_actions:
                self.Q[gs_unique_id][a] = (np.random.random() * 2.0 - 1.0) / 10.0

        if np.random.random() <= self.epsilon:
            chosen_action = np.random.choice(available_actions)
        else:
            chosen_action = max(self.Q[gs_unique_id], key=self.Q[gs_unique_id].get)

        if self.s is not None:
            self.Q[self.s][self.a] += self.alpha * (
                self.r
                + self.gamma * max(self.Q[gs_unique_id].values())
                - self.Q[self.s][self.a]
            )

        self.s = gs_unique_id
        self.a = chosen_action
        self.r = 0.0

        return self.a

    def observe(self, r: float, t: bool, player_index: int):
        if self.r is None:
            return

        self.r += r

        if t:
            self.Q[self.s][self.a] += self.alpha * (self.r - self.Q[self.s][self.a])
            self.s = None
            self.a = None
            self.r = None

    def save_model(self, filename: str):
        with open(f"{filename}_{datetime.now().strftime('%H-%M-%S')}.pkl", "wb",) as f:
            pickle.dump(self.Q, f)

    def load_model(self, filename: str):
        with open(filename, "rb") as f:
            self.Q = pickle.load(f)
