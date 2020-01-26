from drl_gym.agents import RandomAgent
from drl_gym.contracts import Agent, GameState
import numpy as np

from drl_gym.runners import (
    run_for_n_games_and_return_stats,
    run_for_n_games_and_return_max,
)


class RandomRolloutAgent(Agent):
    def __init__(self, epochs_per_action: int, determinist_environment: bool = False):
        self.epochs_per_action = epochs_per_action
        self.determinist_environment = determinist_environment
        self.agents = None

    def act(self, gs: GameState) -> int:
        available_actions = gs.get_available_actions(gs.get_active_player())
        if self.agents is None:
            self.agents = [RandomAgent()] * gs.player_count()
        accumulated_scores = np.zeros((len(available_actions),))

        for i, a in enumerate(available_actions):
            gs_clone = gs.clone()
            gs_clone.step(gs.get_active_player(), a)
            if self.determinist_environment:
                max_scores = run_for_n_games_and_return_max(
                    self.agents, gs_clone, self.epochs_per_action
                )
                accumulated_scores[i] = max_scores[gs.get_active_player()]
            else:
                (total_scores, _, _) = run_for_n_games_and_return_stats(
                    self.agents, gs_clone, self.epochs_per_action
                )
                accumulated_scores[i] = total_scores[gs.get_active_player()]

        # print((accumulated_scores, available_actions[np.argmax(accumulated_scores)]))
        return available_actions[np.argmax(accumulated_scores)]

    def observe(self, r: float, t: bool, player_index: int):
        pass

    def save_model(self, filename: str):
        pass
