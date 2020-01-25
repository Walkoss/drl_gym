import os
import random
from typing import List

import numpy as np

from drl_gym.contracts import GameState


class MinesweeperGameState(GameState):
    def __init__(self):
        self.world = np.zeros([5, 5], dtype=int)
        self.bombs = np.zeros([6, 6], dtype=int)
        self.solution = np.zeros([6, 6], dtype=int)
        for r in range(5):
            for c in range(5):
                if random.random() < 0.2:
                    self.bombs[r][c] = -1
                    self.world[r][c] = -1
        for r in range(5):
            for c in range(5):
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        if self.bombs[rr][cc]:
                            self.solution[r][c] += 1
        for r in range(5):
            for c in range(5):
                if self.world[r][c] != -1:
                    self.world[r][c] = self.solution[r][c]

        self.board = np.array([
            [-2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2],
            [-2, -2, -2, -2, -2],
        ])  # -2 = vide, -1 = bombe, 0 à 8 = nombre de bombes qui entourent
        self.nbr_bombs = np.sum(self.world == -1)
        self.game_over = False
        self.scores = np.array([0], dtype=np.float)
        self.available_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                  24]

    def player_count(self) -> int:
        return 1

    def is_game_over(self) -> bool:
        return self.game_over

    def get_active_player(self) -> int:
        return 0

    def clone(self) -> 'GameState':
        gs_copy = MinesweeperGameState()
        gs_copy.game_over = self.game_over
        gs_copy.scores = self.scores.copy()
        gs_copy.available_actions = self.available_actions.copy()
        gs_copy.nbr_bombs = self.nbr_bombs.copy()
        gs_copy.world = self.world.copy()
        gs_copy.board = self.board.copy()

        return gs_copy

    def step(self, player_index: int, action_index: int):
        assert (not self.game_over)
        assert (player_index == 0)
        assert (action_index in self.available_actions)

        (wanted_i, wanted_j) = (action_index // 5, action_index % 5)

        potential_cell_type = self.world[wanted_i, wanted_j]
        self.board[wanted_i, wanted_j] = potential_cell_type

        self.available_actions.remove(action_index)

        if potential_cell_type == -1:
            # self.board[wanted_i, wanted_j] = '*'
            self.game_over = True
        elif np.sum(self.board == -2) == self.nbr_bombs:  # Fin de jeu -> victoire
            self.game_over = True
            self.scores[player_index] = 25
        else:
            self.scores[player_index] += 1

    def get_scores(self) -> np.ndarray:
        return self.scores

    def get_available_actions(self, player_index: int) -> List[int]:
        return self.available_actions

    def __str__(self):
        str_acc = f"Game Over : {self.game_over}{os.linesep}"
        str_acc += f"Scores : {self.scores}{os.linesep}"

        for i, line in enumerate(self.board):
            for j, cell_type in enumerate(line):
                str_acc += f"{cell_type}"
            str_acc += f"{os.linesep}"

        return str_acc

    def get_unique_id(self) -> int:
        acc = 0
        for i in range(25):
            acc += (5 ** i) * (self.board[i // 5, i % 5] + 1)
        return acc

    def get_max_state_count(self) -> int:
        return 8 ** 25

    def get_action_space_size(self) -> int:
        return len(self.available_actions)

    def get_vectorized_state(self) -> np.ndarray:
        """state_vec = np.zeros(5 * 5 * 8)
        for i in range(5):
            for j in range(5):
                state_vec[i * 5 * 5 + j * 8 + (self.board[i, j] + 1)] = 1
        print("TEST" + state_vec)
        return state_vec"""
        pass

    def render(self):
        print(self)
