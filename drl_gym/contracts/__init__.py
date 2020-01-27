from typing import List, Optional

import numpy as np


class GameState:
    def player_count(self) -> int:
        raise NotImplementedError

    def is_game_over(self) -> bool:
        raise NotImplementedError

    def get_active_player(self) -> int:
        raise NotImplementedError

    def clone(self) -> "GameState":
        raise NotImplementedError

    def step(self, player_index: int, action_index: int):
        raise NotImplementedError

    def get_scores(self) -> np.ndarray:
        raise NotImplementedError

    def get_available_actions(self, player_index: int) -> List[int]:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def get_unique_id(self) -> str:
        raise NotImplementedError

    def get_max_state_count(self) -> int:
        raise NotImplementedError

    def get_action_space_size(self) -> int:
        raise NotImplementedError

    def get_vectorized_state(self, mode: str = None) -> np.ndarray:
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def get_valid_action_from_mouse_pos(
        self, mouse_x: int, mouse_y: int
    ) -> Optional[int]:
        raise NotImplementedError


class Agent:
    def act(self, gs: GameState) -> int:
        raise NotImplementedError

    def observe(self, r: float, t: bool, player_index: int):
        raise NotImplementedError

    def save_model(self, filename: str):
        raise NotImplementedError

    def load_model(self, filename: str):
        raise NotImplementedError
