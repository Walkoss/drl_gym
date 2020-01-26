import os
import random
import hashlib
from typing import List
import numpy as np
from drl_gym.contracts import GameState


class Cell:
    EMPTY = -1
    BOMB = -2


class MinesweeperGameState(GameState):
    def __init__(self, random_map: bool = False):
        self.game_over = False
        self.scores = np.array([0], dtype=int)
        self.available_actions = [i for i in range(81)]
        self.board = np.full((9, 9), Cell.EMPTY, dtype=np.float)

        if random_map:
            self.solution_grid = np.zeros((9, 9), dtype=np.float)
        else:
            self.solution_grid = np.array(
                [
                    [1, 1, 1, 0, 0, 1, 1, 1, 0],
                    [1, -2, 2, 1, 1, 1, -2, 1, 0],
                    [1, 1, 2, -2, 1, 1, 1, 1, 0],
                    [1, 1, 2, 1, 1, 1, 1, 1, 0],
                    [1, -2, 1, 1, -2, 2, -2, 2, 0],
                    [1, 1, 2, 1, 1, 2, 2, -2, 2],
                    [1, 2, -2, 1, 0, 1, 1, 1, 0],
                    [1, -2, 2, 0, 0, 1, -2, 1, 0],
                    [1, 1, 1, 0, 0, 1, 1, 1, 0],
                ],
                dtype=np.float,
            )

        # Pygame
        self.screen = None
        self.font = None

        if random_map:
            for n in range(10):
                self.place_bomb()

            for r in range(9):
                for c in range(9):
                    value = self.solution_grid[r][c]
                    if value == Cell.BOMB:
                        self.update_values(r, c)

    def place_bomb(self):
        r = random.randint(0, 8)
        c = random.randint(0, 8)
        current_row = self.solution_grid[r]
        if not current_row[c] == Cell.BOMB:
            current_row[c] = Cell.BOMB
        else:
            self.place_bomb(self.solution_grid)

    def update_values(self, rn, c):
        if rn - 1 > -1:
            r = self.solution_grid[rn - 1]
            if c - 1 > -1:
                if not r[c - 1] == Cell.BOMB:
                    r[c - 1] += 1
            if not r[c] == Cell.BOMB:
                r[c] += 1
            if 9 > c + 1:
                if not r[c + 1] == Cell.BOMB:
                    r[c + 1] += 1

        r = self.solution_grid[rn]
        if c - 1 > -1:
            if not r[c - 1] == Cell.BOMB:
                r[c - 1] += 1
        if 9 > c + 1:
            if not r[c + 1] == Cell.BOMB:
                r[c + 1] += 1

        if 9 > rn + 1:
            r = self.solution_grid[rn + 1]
            if c - 1 > -1:
                if not r[c - 1] == Cell.BOMB:
                    r[c - 1] += 1
            if not r[c] == Cell.BOMB:
                r[c] += 1
            if 9 > c + 1:
                if not r[c + 1] == Cell.BOMB:
                    r[c + 1] += 1

    def player_count(self) -> int:
        return 1

    def is_game_over(self) -> bool:
        return self.game_over

    def get_active_player(self) -> int:
        return 0

    def clone(self) -> "GameState":
        gs_copy = MinesweeperGameState()
        gs_copy.game_over = self.game_over
        gs_copy.scores = self.scores.copy()
        gs_copy.available_actions = self.available_actions.copy()
        gs_copy.solution_grid = self.solution_grid.copy()
        gs_copy.board = self.board.copy()

        return gs_copy

    def step(self, player_index: int, action_index: int):
        assert not self.game_over
        assert player_index == 0
        assert action_index in self.available_actions

        (wanted_i, wanted_j) = (action_index // 9, action_index % 9)

        potential_cell_type = self.solution_grid[wanted_i, wanted_j]
        self.board[wanted_i, wanted_j] = potential_cell_type

        self.available_actions.remove(action_index)

        if potential_cell_type == Cell.BOMB:
            self.game_over = True
        elif potential_cell_type == 0:
            self.reveal(wanted_i, wanted_j)
        elif np.sum(self.board == Cell.EMPTY) == 10:  # Victoire -> fin de jeu
            self.game_over = True
        else:
            self.scores[player_index] += 1

    def reveal(self, wanted_i, wanted_j):
        neighbors = self.get_neighbors(wanted_i, wanted_j)
        for neighbor_r, neighbor_c in neighbors:
            if (
                self.solution_grid[neighbor_r, neighbor_c] == 0
                and self.board[neighbor_r, neighbor_c] == Cell.EMPTY
            ):
                self.board[neighbor_r, neighbor_c] = 0
                self.available_actions.remove(neighbor_r * 9 + neighbor_c)
                self.reveal(neighbor_r, neighbor_c)

    def get_neighbors(self, wanted_i, wanted_j):
        neighbors = []

        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                elif -1 < (wanted_i + i) < 9 and -1 < (wanted_j + j) < 9:
                    neighbors.append((wanted_i + i, wanted_j + j))

        return neighbors

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
        return hashlib.sha1(self.get_vectorized_state()).hexdigest()

    def get_max_state_count(self) -> int:
        return 11 ** 81  # Nbr état possible d'une case ** (nbr row * nbr columns)

    def get_action_space_size(self) -> int:
        return 81

    def get_vectorized_state(self, mode: str = None) -> np.ndarray:
        if mode:
            return self.board.reshape((9, 9, 1))
        return self.board.reshape(81)

    def render(self):
        print(self)
