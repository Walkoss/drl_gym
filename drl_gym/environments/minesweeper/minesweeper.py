import os
import random
from typing import List

import numpy as np

from drl_gym.contracts import GameState


class MinesweeperGameState(GameState):
    def __init__(self):
        self.game_over = False
        self.scores = np.array([0], dtype=np.float)
        self.available_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                  24]
        self.board = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
        ], dtype=np.float)  # -2 = bombe, -1 = vide, 0 à 8 = nombre de bombes qui entourent
        self.world = np.zeros([8, 8], dtype=np.float)
        for n in range(10):
            self.place_bomb(self.world)

        for r in range(8):
            for c in range(8):
                value = self.l(r, c, self.world)
                if value == -2:
                    self.update_values(r, c, self.world)

    def place_bomb(self, world):
        r = random.randint(0, 7)
        c = random.randint(0, 7)
        current_row = world[r]
        if not current_row[c] == -2:
            current_row[c] = -2
        else:
            self.place_bomb(world)

    def update_values(self, rn, c, world):
        # Row above.
        if rn - 1 > -1:
            r = world[rn - 1]
            if c - 1 > -1:
                if not r[c - 1] == -2:
                    r[c - 1] += 1
            if not r[c] == -2:
                r[c] += 1
            if 8 > c + 1:
                if not r[c + 1] == -2:
                    r[c + 1] += 1

        # Same row.
        r = world[rn]
        if c - 1 > -1:
            if not r[c - 1] == -2:
                r[c - 1] += 1
        if 8 > c + 1:
            if not r[c + 1] == -2:
                r[c + 1] += 1

        # Row below.
        if 8 > rn + 1:
            r = world[rn + 1]
            if c - 1 > -1:
                if not r[c - 1] == -2:
                    r[c - 1] += 1
            if not r[c] == -2:
                r[c] += 1
            if 8 > c + 1:
                if not r[c + 1] == -2:
                    r[c + 1] += 1

    def l(self, r, c, world):
        row = world[r]
        c = row[c]
        return c

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

        if potential_cell_type == -2:
            self.game_over = True
        elif np.sum(self.board == -1) == 10:  # Victoire -> fin de jeu
            self.game_over = True
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
        return 11 ** (8 * 8)  # Nbr état possible d'une case ** (nbr row * nbr columns)

    def get_action_space_size(self) -> int:
        return len(self.available_actions)

    def get_vectorized_state(self) -> np.ndarray:
        """state_vec = np.zeros(5 * 5 * 8)
        for i in range(5):
            for j in range(5):
                state_vec[i * 5 * 5 + j * 8 + (self.board[i, j] + 1)] = 1
        return state_vec"""
        pass

    def render(self):
        print(self)
