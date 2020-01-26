import os
import random
import hashlib
from typing import List
import pygame
import numpy as np
from drl_gym.contracts import GameState

pygame.init()
COLUMNS = 9
ROWS = 9
SQUARE_SIZE = 100
WIDTH = COLUMNS * SQUARE_SIZE
HEIGHT = (ROWS + 1) * SQUARE_SIZE
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
DRAWING_DELAY = 500


class Cell:
    EMPTY = -1
    BOMB = -2


class MinesweeperGameState(GameState):
    def __init__(self):
        self.game_over = False
        self.scores = np.array([0], dtype=int)
        self.available_actions = [i for i in range(81)]
        self.board = np.full((9, 9), Cell.EMPTY, dtype=np.float)  # -2 = bombe, -1 = vide, 0 à 8 = nombre de bombes qui entourent
        self.world = np.zeros([9, 9], dtype=np.float)

        # Pygame
        self.screen = None
        self.font = None

        for n in range(10):
            self.place_bomb(self.world)

        for r in range(9):
            for c in range(9):
                value = self.l(r, c, self.world)
                if value == -2:
                    self.update_values(r, c, self.world)

    def place_bomb(self, world):
        r = random.randint(0, 8)
        c = random.randint(0, 8)
        current_row = world[r]
        if not current_row[c] == Cell.BOMB:
            current_row[c] = Cell.BOMB
        else:
            self.place_bomb(world)

    def update_values(self, rn, c, world):
        # Row above.
        if rn - 1 > -1:
            r = world[rn - 1]
            if c - 1 > -1:
                if not r[c - 1] == Cell.BOMB:
                    r[c - 1] += 1
            if not r[c] == -2:
                r[c] += 1
            if 9 > c + 1:
                if not r[c + 1] == Cell.BOMB:
                    r[c + 1] += 1

        # Same row.
        r = world[rn]
        if c - 1 > -1:
            if not r[c - 1] == Cell.BOMB:
                r[c - 1] += 1
        if 9 > c + 1:
            if not r[c + 1] == Cell.BOMB:
                r[c + 1] += 1

        # Row below.
        if 9 > rn + 1:
            r = world[rn + 1]
            if c - 1 > -1:
                if not r[c - 1] == Cell.BOMB:
                    r[c - 1] += 1
            if not r[c] == Cell.BOMB:
                r[c] += 1
            if 9 > c + 1:
                if not r[c + 1] == Cell.BOMB:
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

    def clone(self) -> "GameState":
        gs_copy = MinesweeperGameState()
        gs_copy.game_over = self.game_over
        gs_copy.scores = self.scores.copy()
        gs_copy.available_actions = self.available_actions.copy()
        gs_copy.world = self.world.copy()
        gs_copy.board = self.board.copy()

        return gs_copy

    def step(self, player_index: int, action_index: int):
        assert not self.game_over
        assert player_index == 0
        assert action_index in self.available_actions

        (wanted_i, wanted_j) = (action_index // 9, action_index % 9)

        potential_cell_type = self.world[wanted_i, wanted_j]
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
        if self.check_neighbors(wanted_i, wanted_j):
            self.board[wanted_i][wanted_j] = self.world[wanted_i][wanted_j]

    def check_neighbors(self, wanted_i, wanted_j):
        return self.world[wanted_i][wanted_j] == 0

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
        return len(self.available_actions)

    def get_vectorized_state(self, mode: str = None) -> np.ndarray:
        return self.board.reshape(81)

    def render(self):
        """if not self.screen and not self.font:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.font = pygame.font.SysFont("monospace", 75)
        if self.game_over:
            label = self.font.render(f"Fin de jeu!", 1, BLACK)
            self.screen.blit(label, (10, 10))
        for c in range(COLUMNS):
            for r in range(ROWS):
                pygame.draw.rect(
                    self.screen,
                    BLUE,
                    (
                        c * SQUARE_SIZE,
                        r * SQUARE_SIZE + SQUARE_SIZE,
                        SQUARE_SIZE,
                        SQUARE_SIZE,
                    ),
                )
                pygame.draw.rect(
                    self.screen,
                    BLACK,
                    (
                        int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                        int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE / 2),
                    ),
                )

        for c in range(COLUMNS):
            for r in range(ROWS):
                if self.board[r][c] == Cell.RED:
                    pygame.draw.rect(
                        self.screen,
                        RED,
                        (
                            int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                            int(r * SQUARE_SIZE + SQUARE_SIZE / 2) + SQUARE_SIZE,
                        ),
                    )
                elif self.board[r][c] == Cell.YELLOW:
                    pygame.draw.rect(
                        self.screen,
                        YELLOW,
                        (
                            int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                            int(r * SQUARE_SIZE + SQUARE_SIZE / 2) + SQUARE_SIZE,
                        ),
                    )
        pygame.display.update()
        pygame.time.delay(DRAWING_DELAY)"""
        print(self)
