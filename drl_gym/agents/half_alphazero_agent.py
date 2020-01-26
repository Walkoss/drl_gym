from math import sqrt, log
from random import choice

import numpy as np
from tensorflow.keras.utils import to_categorical

from drl_gym.brains import AlphaQNetwork
from drl_gym.contracts import Agent, GameState


class HalfAlphaZeroAgent(Agent):
    def __init__(
        self, max_iteration: int, action_space_size: int, keep_memory: bool = True
    ):
        self.max_iteration = max_iteration
        self.keep_memory = keep_memory
        self.memory = dict()
        self.action_space_size = action_space_size
        self.brain = AlphaQNetwork(
            output_dim=action_space_size,
            hidden_layers_count=5,
            neurons_per_hidden_layer=64,
        )

        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.intermediate_reward = 0.0

        self.states_batch = None
        self.actions_batch = None
        self.gains_batch = None

    @staticmethod
    def create_node_in_memory(
        memory, node_hash, available_actions, current_player, q_values
    ):
        memory[node_hash] = [
            {"r": 0, "n": 0, "np": 0, "a": a, "p": current_player, "q": q_values[a]}
            for a in available_actions
        ]

    @staticmethod
    def ucb_1(edge):
        return edge["r"] / edge["n"] + sqrt(2 * log(edge["np"]) / edge["n"])

    def act(self, gs: GameState) -> int:
        root_hash = gs.get_unique_id()
        memory = self.memory if self.keep_memory else dict()

        if root_hash not in memory:
            q_values = self.brain.predict(gs.get_vectorized_state())
            HalfAlphaZeroAgent.create_node_in_memory(
                memory,
                root_hash,
                gs.get_available_actions(gs.get_active_player()),
                gs.get_active_player(),
                q_values,
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
                    ((edge, HalfAlphaZeroAgent.ucb_1(edge)) for edge in memory[s]),
                    key=lambda kv: kv[1],
                )[0]
                history.append((s, chosen_edge))

                gs_copy.step(gs_copy.get_active_player(), chosen_edge["a"])
                s = gs_copy.get_unique_id()
                if s not in memory:
                    q_values = self.brain.predict(gs_copy.get_vectorized_state())
                    HalfAlphaZeroAgent.create_node_in_memory(
                        memory,
                        s,
                        gs_copy.get_available_actions(gs_copy.get_active_player()),
                        gs_copy.get_active_player(),
                        q_values,
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
                    q_values = self.brain.predict(gs_copy.get_vectorized_state())
                    HalfAlphaZeroAgent.create_node_in_memory(
                        memory,
                        s,
                        gs_copy.get_available_actions(gs_copy.get_active_player()),
                        gs_copy.get_active_player(),
                        q_values,
                    )

            scores = np.zeros(gs_copy.player_count())
            scores_set = np.zeros(gs_copy.player_count())
            # REMONTEE DU SCORE
            for (s, edge) in history:
                if scores_set[edge["p"]] == 0:
                    scores_set[edge["p"]] = 1.0
                    scores[edge["p"]] = edge["q"]

                edge["n"] += 1
                edge["r"] += scores[edge["p"]]
                for neighbour_edge in memory[s]:
                    neighbour_edge["np"] += 1

        chosen_action = max((edge for edge in memory[root_hash]), key=lambda e: e["n"])[
            "a"
        ]

        if len(self.states_buffer) > 0:
            self.rewards_buffer.append(self.intermediate_reward)

        self.states_buffer.append(gs.get_vectorized_state())
        self.actions_buffer.append(
            to_categorical(chosen_action, gs.get_action_space_size())
        )
        self.intermediate_reward = 0.0

        return chosen_action

    def observe(self, r: float, t: bool, player_index: int):
        if len(self.states_buffer) == 0:
            return

        self.intermediate_reward += r

        if t:
            self.rewards_buffer.append(self.intermediate_reward)

            states = np.array(self.states_buffer)
            chosen_actions = np.array(self.actions_buffer)
            rewards = np.array(self.rewards_buffer)
            gains = np.zeros_like(rewards)

            previous_gain = 0.0
            for i in reversed(range(len(rewards))):
                gains[i] = rewards[i] + 0.99 * previous_gain

            self.states_buffer = []
            self.actions_buffer = []
            self.rewards_buffer = []
            self.intermediate_reward = 0.0

            self.states_batch = (
                np.concatenate((self.states_batch, states))
                if self.states_batch is not None
                else states
            )
            self.actions_batch = (
                np.concatenate((self.actions_batch, chosen_actions))
                if self.actions_batch is not None
                else chosen_actions
            )
            self.gains_batch = (
                np.concatenate((self.gains_batch, gains))
                if self.gains_batch is not None
                else gains
            )

            if len(self.states_batch) > 1024:
                self.brain.train(
                    self.states_batch, self.actions_batch, self.gains_batch
                )
                self.states_batch = None
                self.actions_batch = None
                self.gains_batch = None
                self.memory = dict()

    def save_model(self, filename: str):
        self.brain.save_model(filename)
