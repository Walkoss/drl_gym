import os
import random
import hashlib
from typing import List, Optional
import numpy as np
import pygame

from drl_gym.contracts import GameState


WIDTH = 900
HEIGHT = 1000
DRAWING_DELAY = 500

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREY = (127, 127, 127)
BLACK = (0, 0, 0)


class Cell:
    EMPTY = -1
    BOMB = -2


class MinesweeperGameState(GameState):
    def __init__(self, random_map: bool = False):
        self.game_over = False
        self.scores = np.array([0], dtype=int)
        self.available_actions = [i for i in range(81)]
        self.board = np.full((9, 9), Cell.EMPTY, dtype=np.float)
        self.has_win = False

        if random_map:
            self.solution_grid = np.zeros((9, 9), dtype=np.float)
        else:
            self.solution_grid = np.array(
                [
                    [1, 1, 0, 0, 1, 1, 1, 0, 0],
                    [-2, 1, 0, 1, 2, -2, 1, 0, 0],
                    [1, 1, 0, 1, -2, 3, 2, 0, 0],
                    [0, 0, 1, 2, 3, -2, 1, 0, 0],
                    [0, 0, 1, -2, 2, 1, 1, 0, 0],
                    [0, 1, 2, 2, 1, 0, 1, 1, 1],
                    [0, 1, -2, 1, 0, 0, 1, -2, 2],
                    [1, 2, 2, 1, 0, 1, 2, 3, -2],
                    [1, -2, 1, 0, 0, 1, -2, 2, 1],
                ],
                dtype=np.float,
            )

        if random_map:
            for n in range(10):
                self.place_bomb()

            for r in range(9):
                for c in range(9):
                    value = self.solution_grid[r][c]
                    if value == Cell.BOMB:
                        self.update_values(r, c)

        # Pygame
        self.screen = None
        self.font = None

    def place_bomb(self):
        r = random.randint(0, 8)
        c = random.randint(0, 8)
        current_row = self.solution_grid[r]
        if not current_row[c] == Cell.BOMB:
            current_row[c] = Cell.BOMB
        else:
            self.place_bomb()

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
            self.scores[player_index] += 1
            self.reveal(wanted_i, wanted_j)
        else:
            self.scores[player_index] += 1
        if np.sum(self.board == Cell.EMPTY) == 10:  # Victoire -> fin de jeu
            self.has_win = True
            self.game_over = True

    def reveal(self, wanted_i, wanted_j):
        neighbors = self.get_neighbors(wanted_i, wanted_j)
        for neighbor_r, neighbor_c in neighbors:
            if (
                self.solution_grid[neighbor_r, neighbor_c] == 0
                and self.board[neighbor_r, neighbor_c] == Cell.EMPTY
            ):
                self.board[neighbor_r, neighbor_c] = 0
                self.step(self.get_active_player(), neighbor_r * 9 + neighbor_c)

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
        if not self.screen and not self.font:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.font = pygame.font.SysFont("monospace", 20)
            pygame.display.set_caption("Minesweeper")
        self.screen.fill(WHITE)

        pygame.draw.rect(self.screen, GREY, (0, 0, WIDTH, 100))
        pygame.draw.line(self.screen, BLACK, (0, 100), (WIDTH, 100), 4)

        if not self.game_over:
            text = self.font.render(
                f"Scores: {self.scores[self.get_active_player()]}", True, BLACK
            )
            text_x = text.get_rect().width
            text_y = text.get_rect().height
            self.screen.blit(text, (((WIDTH / 2) - (text_x / 2)), (50 - (text_y / 2))))
        elif self.has_win:  # win
            text = self.font.render("You win", True, BLACK)
            text_x = text.get_rect().width
            text_y = text.get_rect().height
            self.screen.blit(text, (((WIDTH / 2) - (text_x / 2)), (50 - (text_y / 2))))
        else:  # loose
            text = self.font.render("You lose", True, BLACK)
            text_x = text.get_rect().width
            text_y = text.get_rect().height
            self.screen.blit(text, (((WIDTH / 2) - (text_x / 2)), (50 - (text_y / 2))))

        for y in range(9):
            for x in range(9):
                if self.board[y][x] != Cell.EMPTY:
                    if self.board[y][x] == Cell.BOMB:
                        pygame.draw.rect(
                            self.screen,
                            RED,
                            (
                                x * (WIDTH / 9),
                                (y * ((HEIGHT - 100) / 9)) + 100,
                                (WIDTH / 9),
                                ((HEIGHT - 100) / 9),
                            ),
                        )
                    else:
                        pygame.draw.rect(
                            self.screen,
                            GREY,
                            (
                                x * (WIDTH / 9),
                                (y * ((HEIGHT - 100) / 9)) + 100,
                                (WIDTH / 9),
                                ((HEIGHT - 100) / 9),
                            ),
                        )
                        text = self.font.render(str(int(self.board[y][x])), True, BLACK)
                        text_x = text.get_rect().width
                        text_y = text.get_rect().height
                        self.screen.blit(
                            text,
                            (
                                (x * (WIDTH / 9) + ((WIDTH / 9) / 2) - (text_x / 2)),
                                (
                                    (y * ((HEIGHT - 100) / 9))
                                    + 100
                                    + (((HEIGHT - 100) / 9) / 2)
                                    - (text_y / 2)
                                ),
                            ),
                        )
                pygame.draw.rect(
                    self.screen,
                    BLACK,
                    (
                        x * (WIDTH / 9),
                        (y * ((HEIGHT - 100) / 9)) + 100,
                        WIDTH / 9,
                        (HEIGHT - 100) / 9,
                    ),
                    1,
                )
        pygame.display.update()
        pygame.time.delay(DRAWING_DELAY)

    def get_valid_action_from_mouse_pos(
        self, mouse_x: int, mouse_y: int
    ) -> Optional[int]:
        return ((mouse_y - 100) // 100) * 9 + (mouse_x // 100)
