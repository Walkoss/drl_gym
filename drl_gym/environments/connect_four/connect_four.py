import hashlib
import os

from typing import List

import numpy as np
import pygame

from drl_gym.contracts import GameState


COLUMNS = 7
ROWS = 6
SQUARE_SIZE = 100
WIDTH = COLUMNS * SQUARE_SIZE
HEIGHT = (ROWS + 1) * SQUARE_SIZE
RADIUS = int(SQUARE_SIZE / 2 - 5)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
DRAWING_DELAY = 500


class Cell:
    EMPTY = -1
    RED = 0
    YELLOW = 1


class ConnectFourGameState(GameState):
    def __init__(self):
        self.board = np.full((ROWS, COLUMNS), Cell.EMPTY, dtype=float)
        self.active_player = Cell.RED
        self.game_over = False
        self.remaining_cells = ROWS * COLUMNS
        self.scores = np.zeros(2)
        self.winner = False

        # Pygame
        self.screen = None
        self.font = None

    def player_count(self) -> int:
        return 2

    def is_game_over(self) -> bool:
        return self.game_over

    def get_active_player(self) -> int:
        return self.active_player

    def clone(self) -> "GameState":
        gs_clone = ConnectFourGameState()

        gs_clone.game_over = self.game_over
        gs_clone.active_player = self.active_player
        gs_clone.board = self.board.copy()
        gs_clone.remaining_cells = self.remaining_cells
        gs_clone.scores = self.scores.copy()
        gs_clone.winner = self.winner

        return gs_clone

    def step(self, player_index: int, action_index: int):
        assert 0 <= action_index < 7, "Invalid action"
        assert player_index == self.active_player, f"Not valid player_index"

        column = self.board[:, action_index]
        column_reversed = column[::-1]

        # Check if first element is empty
        if column[0] != Cell.EMPTY:
            raise ValueError(f"Cannot insert at column index {action_index}")

        y = len(column) - np.argmin(column_reversed) - 1
        column[y] = self.active_player
        self.remaining_cells += -1

        # Check for win or tie
        if self.check_for_win(row_idx=y, column_idx=action_index):
            self.scores[player_index] = 1
            self.scores[(player_index + 1) % 2] = -1
            self.winner = "Red" if player_index == Cell.RED else "Yellow"
            # print(f"{self.winner} player won")
            self.game_over = True
            return
        elif self.remaining_cells == 0:
            # print("Tie")
            # scores to 0.5 for each player ?
            self.game_over = True
            return

        # Switch player
        self.active_player = Cell.YELLOW if self.active_player == Cell.RED else Cell.RED

    def get_scores(self) -> np.ndarray:
        return self.scores

    def get_available_actions(self, player_index: int) -> List[int]:
        available_actions = []
        for i, column in enumerate(self.board.T):
            if Cell.EMPTY in column:
                available_actions.append(i)
        return available_actions

    def __str__(self):
        def _get_char_from_cell_type(cell: Cell) -> str:
            if cell == Cell.EMPTY:
                return " "
            elif cell == Cell.YELLOW:
                return "Y"
            elif cell == Cell.RED:
                return "R"

        str_acc = (
            f"--------------------------{os.linesep}"
            f"0 | 1 | 2 | 3 | 4 | 5 | 6{os.linesep}"
        )

        return (
            f"{os.linesep}".join(
                [
                    " |\t".join([_get_char_from_cell_type(cell) for cell in row])
                    for row in self.board
                ]
            )
            + os.linesep
            + str_acc
        )

    def get_unique_id(self) -> str:
        return hashlib.sha1(self.get_vectorized_state()).hexdigest()

    def get_max_state_count(self) -> int:
        return 3 ** 42

    def get_action_space_size(self) -> int:
        return 7

    def get_vectorized_state(self, mode: str = None) -> np.ndarray:
        if mode == "2D":
            return self.board.reshape((6, 7, 1))
        return self.board.reshape(42)

    def check_for_win_on_line(self, line) -> bool:
        if len(line) < 4:
            return False

        consecutive_count = 0
        old_cell = None

        for current_cell in line:
            if current_cell != Cell.EMPTY:
                if old_cell is None:
                    old_cell = current_cell
                    consecutive_count += 1
                elif old_cell == current_cell:
                    consecutive_count += 1
                    if consecutive_count == 4:
                        return True
                elif old_cell != current_cell:
                    old_cell = current_cell
                    consecutive_count = 0
            else:
                consecutive_count = 0
                old_cell = None

        return False

    def check_for_win(self, row_idx, column_idx) -> bool:
        # Check row
        row = self.board[row_idx]
        if self.check_for_win_on_line(row):
            return True

        # Check column
        column = self.board[:, column_idx]
        if self.check_for_win_on_line(column):
            return True

        # Check diagonals
        flip_board = np.fliplr(self.board)
        if row_idx > column_idx:
            first_diag = np.diag(self.board, k=-row_idx + column_idx)
            second_diag = np.diag(flip_board, k=-row_idx + (6 - column_idx))
        elif row_idx == column_idx:
            first_diag = np.diag(self.board)
            second_diag = np.diag(flip_board, k=6 - (row_idx + column_idx))
        else:
            first_diag = np.diag(self.board, k=column_idx - row_idx)
            second_diag = np.diag(flip_board, k=(6 - column_idx) - row_idx)

        if self.check_for_win_on_line(first_diag):
            return True
        if self.check_for_win_on_line(second_diag):
            return True

        return False

    def render(self):
        if not self.screen and not self.font:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.font = pygame.font.SysFont("monospace", 50)
            pygame.display.set_caption("Connect four")
        self.screen.fill(WHITE)

        if self.game_over:
            if self.winner:
                text = self.font.render(f"{self.winner} wins!", True, BLACK)
                text_x = text.get_rect().width
                text_y = text.get_rect().height
                self.screen.blit(
                    text, (((WIDTH / 2) - (text_x / 2)), (50 - (text_y / 2)))
                )
            else:
                text = self.font.render(f"Tie!", True, BLACK)
                text_x = text.get_rect().width
                text_y = text.get_rect().height
                self.screen.blit(
                    text, (((WIDTH / 2) - (text_x / 2)), (50 - (text_y / 2)))
                )
        else:
            if self.active_player == Cell.RED:
                pygame.draw.rect(
                    self.screen, RED, (0, 0, WIDTH, HEIGHT),
                )
            elif self.active_player == Cell.YELLOW:
                pygame.draw.rect(
                    self.screen, YELLOW, (0, 0, WIDTH, HEIGHT),
                )
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
                pygame.draw.circle(
                    self.screen,
                    WHITE,
                    (
                        int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                        int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE / 2),
                    ),
                    RADIUS,
                )

        for c in range(COLUMNS):
            for r in range(ROWS):
                if self.board[r][c] == Cell.RED:
                    pygame.draw.circle(
                        self.screen,
                        RED,
                        (
                            int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                            int(r * SQUARE_SIZE + SQUARE_SIZE / 2) + SQUARE_SIZE,
                        ),
                        RADIUS,
                    )
                elif self.board[r][c] == Cell.YELLOW:
                    pygame.draw.circle(
                        self.screen,
                        YELLOW,
                        (
                            int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                            int(r * SQUARE_SIZE + SQUARE_SIZE / 2) + SQUARE_SIZE,
                        ),
                        RADIUS,
                    )
        pygame.display.update()
        pygame.time.delay(DRAWING_DELAY)
