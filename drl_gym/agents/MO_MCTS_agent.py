import pickle
from math import sqrt, log
from random import choice

from drl_gym.contracts import Agent, GameState


class MOMCTSAgent(Agent):
    def __init__(self, max_iteration: int, keep_memory: bool = True):
        self.max_iteration = max_iteration
        self.keep_memory = keep_memory
        self.memory = dict()

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
        root_hash = gs.get_unique_id()
        memory = self.memory if self.keep_memory else dict()

        if root_hash not in memory:
            MOMCTSAgent.create_node_in_memory(
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
                    ((edge, MOMCTSAgent.ucb_1(edge)) for edge in memory[s]),
                    key=lambda kv: kv[1],
                )[0]
                history.append((s, chosen_edge))

                gs_copy.step(gs_copy.get_active_player(), chosen_edge["a"])
                s = gs_copy.get_unique_id()
                if s not in memory:
                    MOMCTSAgent.create_node_in_memory(
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
                    MOMCTSAgent.create_node_in_memory(
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

        return max((edge for edge in memory[root_hash]), key=lambda e: e["n"])["a"]

    def observe(self, r: float, t: bool, player_index: int):
        pass

    def save_model(self, filename: str):
        with open(f"{filename}.pkl", "wb",) as f:
            pickle.dump(self.memory, f)

    def load_model(self, filename: str):
        with open(filename, "rb") as f:
            self.memory = pickle.load(f)
