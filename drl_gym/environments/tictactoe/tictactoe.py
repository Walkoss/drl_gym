from typing import List

import os

import numpy as np

from drl_gym.contracts import GameState


class TicTacToeGameState(GameState):
    def __init__(self):
        self.game_over = False
        self.active_player = 0
        self.scores = np.zeros(2)
        self.available_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.board = np.ones((3, 3), dtype=np.int) * -1

    def player_count(self) -> int:
        return 2

    def is_game_over(self) -> bool:
        return self.game_over

    def get_active_player(self) -> int:
        return self.active_player

    def clone(self) -> "GameState":
        gs_clone = TicTacToeGameState()
        gs_clone.game_over = self.game_over
        gs_clone.active_player = self.active_player
        gs_clone.scores = self.scores.copy()
        gs_clone.available_actions = self.available_actions.copy()
        gs_clone.board = self.board.copy()
        return gs_clone

    def step(self, player_index: int, action_index: int):
        assert not self.game_over
        assert player_index == self.active_player
        assert 0 <= action_index <= 8

        (wanted_i, wanted_j) = (action_index // 3, action_index % 3)

        potential_cell_type = self.board[wanted_i, wanted_j]
        assert potential_cell_type == -1

        self.board[wanted_i, wanted_j] = player_index
        self.available_actions.remove(action_index)

        if (
            self.board[wanted_i, 0]
            == self.board[wanted_i, 1]
            == self.board[wanted_i, 2]
            or self.board[0, wanted_j]
            == self.board[1, wanted_j]
            == self.board[2, wanted_j]
            or self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == player_index
            or self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == player_index
        ):
            self.game_over = True
            self.scores[player_index] = 1
            self.scores[(player_index + 1) % 2] = -1
            return

        for i in range(3):
            for j in range(3):
                if self.board[i, j] == -1:
                    self.active_player = (self.active_player + 1) % 2
                    return

        self.game_over = True
        return

    def get_scores(self) -> np.ndarray:
        return self.scores

    def get_available_actions(self, player_index: int) -> List[int]:
        return self.available_actions

    def __str__(self):
        str_acc = f"Game Over : {self.game_over}{os.linesep}"
        str_acc += f"Remaining Actions : {self.available_actions}{os.linesep}"
        str_acc += f"Scores : {self.scores}{os.linesep}"

        for i, line in enumerate(self.board):
            for j, cell_type in enumerate(line):
                if cell_type == -1:
                    str_acc += " "
                elif cell_type == 0:
                    str_acc += "X"
                else:
                    str_acc += "O"
            str_acc += f"{os.linesep}"

        return str_acc

    def get_unique_id(self) -> str:
        acc = 0
        for i in range(9):
            acc += (3 ** i) * (self.board[i // 3, i % 3] + 1)
        return str(acc)

    def get_max_state_count(self) -> int:
        return 3 ** 9

    def get_action_space_size(self) -> int:
        return 9

    def get_vectorized_state(self, mode: str = None) -> np.ndarray:
        state_vec = np.zeros(3 * 3 * 3)
        for i in range(3):
            for j in range(3):
                state_vec[i * 3 * 3 + j * 3 + (self.board[i, j] + 1)] = 1
        return state_vec

    def render(self):
        print(self)


if __name__ == "__main__":
    gs = TicTacToeGameState()
    print(gs)
