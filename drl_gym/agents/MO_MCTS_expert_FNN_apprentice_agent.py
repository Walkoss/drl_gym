from math import sqrt, log
from random import choice

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam

from drl_gym.contracts import Agent, GameState


class ExpertApprenticeAgent(Agent):
    def __init__(
        self,
        max_iteration: int,
        action_space_size: int,
        keep_memory: bool = True,
        apprentice_training_before_takeover: int = 100,
    ):
        self.max_iteration = max_iteration
        self.keep_memory = keep_memory
        self.memory = dict()
        self.brain = Sequential()
        self.brain.add(Dense(64, activation=relu))
        self.brain.add(Dense(64, activation=relu))
        self.brain.add(Dense(64, activation=relu))
        self.brain.add(Dense(action_space_size, activation=softmax))
        self.brain.compile(optimizer=Adam(), loss=mse)
        self.apprentice_training_before_takeover = apprentice_training_before_takeover
        self.apprentice_training_count = 0

        self.states_buffer = []
        self.actions_buffer = []

    @staticmethod
    def create_node_in_memory(memory, node_hash, available_actions, current_player):
        memory[node_hash] = [
            {"r": 0, "n": 0, "np": 0, "a": a, "p": current_player}
            for a in available_actions
        ]

    @staticmethod
    def ucb_1(edge):
        return edge["r"] / edge["n"] + sqrt(2 * log(edge["np"]) / edge["n"])

    def act(self, gs: GameState) -> int:

        if self.apprentice_training_count > self.apprentice_training_before_takeover:
            return gs.get_available_actions(gs.get_active_player())[
                np.argmax(
                    self.brain.predict(np.array([gs.get_vectorized_state()]))[0][
                        gs.get_available_actions(gs.get_active_player())
                    ]
                )
            ]

        root_hash = gs.get_unique_id()
        memory = self.memory if self.keep_memory else dict()

        if root_hash not in memory:
            ExpertApprenticeAgent.create_node_in_memory(
                memory,
                root_hash,
                gs.get_available_actions(gs.get_active_player()),
                gs.get_active_player(),
            )

        for i in range(self.max_iteration):
            gs_copy = gs.clone()
            s = gs_copy.get_unique_id()
            history = []

            # SELECTION
            while not gs_copy.is_game_over() and all(
                (edge["n"] > 0 for edge in memory[s])
            ):
                chosen_edge = max(
                    ((edge, ExpertApprenticeAgent.ucb_1(edge)) for edge in memory[s]),
                    key=lambda kv: kv[1],
                )[0]
                history.append((s, chosen_edge))

                gs_copy.step(gs_copy.get_active_player(), chosen_edge["a"])
                s = gs_copy.get_unique_id()
                if s not in memory:
                    ExpertApprenticeAgent.create_node_in_memory(
                        memory,
                        s,
                        gs_copy.get_available_actions(gs_copy.get_active_player()),
                        gs_copy.get_active_player(),
                    )

            # EXPANSION
            if not gs_copy.is_game_over():
                chosen_edge = choice(
                    list(filter(lambda e: e["n"] == 0, (edge for edge in memory[s])))
                )

                history.append((s, chosen_edge))
                gs_copy.step(gs_copy.get_active_player(), chosen_edge["a"])
                s = gs_copy.get_unique_id()
                if s not in memory:
                    ExpertApprenticeAgent.create_node_in_memory(
                        memory,
                        s,
                        gs_copy.get_available_actions(gs_copy.get_active_player()),
                        gs_copy.get_active_player(),
                    )

            # SIMULATION
            while not gs_copy.is_game_over():
                gs_copy.step(
                    gs_copy.get_active_player(),
                    choice(gs_copy.get_available_actions(gs_copy.get_active_player())),
                )

            scores = gs_copy.get_scores()
            # REMONTEE DU SCORE
            for (s, edge) in history:
                edge["n"] += 1
                edge["r"] += scores[edge["p"]]
                for neighbour_edge in memory[s]:
                    neighbour_edge["np"] += 1

        target = np.zeros(gs.get_action_space_size())

        for edge in memory[root_hash]:
            target[edge["a"]] = edge["n"]

        target /= np.sum(target)

        self.states_buffer.append(gs.get_vectorized_state())
        self.actions_buffer.append(target)

        if len(self.states_buffer) > 200:
            self.apprentice_training_count += 1
            self.brain.fit(
                np.array(self.states_buffer), np.array(self.actions_buffer), verbose=0
            )
            self.states_buffer.clear()
            self.actions_buffer.clear()

        if self.apprentice_training_count > self.apprentice_training_before_takeover:
            print("Apprentice is playing next round")

        return max((edge for edge in memory[root_hash]), key=lambda e: e["n"])["a"]

    def observe(self, r: float, t: bool, player_index: int):
        pass

    def save_model(self, filename: str):
        self.brain.save(f"{filename}.h5")
