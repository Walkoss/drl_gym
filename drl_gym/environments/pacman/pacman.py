import hashlib
import os
import numpy as np
import pygame

from typing import List
from drl_gym.contracts import GameState

# Movement per frame
SPEED = 0.4

# Size in pixel
SPRITE_SIZE = 0.5

# Game rules
STARTING_HEART = 2
PACMAN_SPAWN_POINT = np.array([23.0, 12], dtype=np.float)
PACMAN_SPAWN_DIRECTION = 2
GHOST_SPAWN_POINT = np.array([11.0, 14.0], dtype=np.float)
GHOST_SPAWN_DIRECTION = 4
GHOST_COUNT = 4
GHOST_POINT = 200
PAC_DOTS_POINT = 10
ENERGIZER_POINT = 50

# Pygame
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (25, 25, 166)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
TUMBLEWEED = (222, 161, 133)
PINK = (255, 184, 255)
AQUA = (0, 255, 255)
ORANGE = (255, 184, 82)
GHOST_COLOR = [RED, AQUA, ORANGE, PINK]
SQUARE_SIZE = 16
WIDTH = 28 * SQUARE_SIZE
HEIGHT = (31 * SQUARE_SIZE) + SQUARE_SIZE * 3


class PacManGameState(GameState):
    def __init__(self):
        # Pyagme
        self.screen = None
        self.font = None

        # 0 = Empty
        # 1 = Wall
        # 2 = Pac-Dots
        # 3 = Energizer
        # 4 = Ghost Portal
        # 5 = Pacman
        # 6 = Ghost
        # fmt: off
        self.maze = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
                [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
                [1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1],
                [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
                [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
                [1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1],
                [1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1],
                [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1],
                [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 4, 4, 1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
                [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
                [1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1],
                [1, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 3, 1],
                [1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1],
                [1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1],
                [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
                [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
                [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        # fmt: on
        self.game_over = False
        self.pacman_pos = PACMAN_SPAWN_POINT
        self.ghost_pos = np.array([GHOST_SPAWN_POINT for _ in range(GHOST_COUNT)])
        self.ghost_direction = [GHOST_SPAWN_DIRECTION for _ in range(GHOST_COUNT)]
        self.pacman_direction = PACMAN_SPAWN_DIRECTION
        self.heart = STARTING_HEART
        self.score = np.array([0], dtype=np.float)
        # UP = 0
        # DOWN = 1
        # LEFT = 2
        # RIGHT = 3
        # NOOP = 4
        self.available_actions = [0, 1, 2, 3, 4]
        self.remaining_actions = 2000
        self.action_vector = {
            0: np.array([-SPEED, 0]),
            1: np.array([SPEED, 0]),
            2: np.array([0, -SPEED]),
            3: np.array([0, SPEED]),
            4: np.array([0, 0]),
        }

    def player_count(self) -> int:
        return 1

    def is_game_over(self) -> bool:
        return self.game_over

    def get_active_player(self) -> int:
        return 0

    def clone(self) -> "GameState":
        gs_copy = PacManGameState()
        gs_copy.game_over = self.game_over
        gs_copy.score = self.score.copy()
        gs_copy.heart = self.heart
        gs_copy.remaining_actions = self.remaining_actions
        gs_copy.pacman_pos = self.pacman_pos.copy()
        gs_copy.pacman_direction = self.pacman_direction
        gs_copy.ghost_pos = self.ghost_pos.copy()
        gs_copy.ghost_direction = self.ghost_direction.copy()
        gs_copy.maze = self.maze.copy()
        return gs_copy

    # TODO: Fix collision issue
    def step(self, player_index: int, action_index: int):
        assert not self.game_over, f"Game is over!"
        assert player_index == self.get_active_player(), f"Not valid player_index!"
        assert 0 <= action_index < self.get_action_space_size(), f"Invalid action!"

        target_pos = self.direction_to_target_position(
            action_index, self.pacman_pos, self.pacman_direction
        )

        assert 0 <= target_pos[0] < self.maze.shape[0]
        assert 0 <= target_pos[1] < self.maze.shape[1]

        target_type = self.evaluate_target_type(
            target_pos, action_index, self.pacman_direction, 5
        )

        self.remaining_actions -= 1
        # Pacman position update
        if target_type == 5:
            # Pacman didn't reach a newer cell
            self.pacman_pos = target_pos
        # Maze player position
        else:
            approximate_player_pos = np.floor(self.pacman_pos).astype(int)
            # Pacman reach a empty cell
            if target_type == 0:
                self.pacman_pos = target_pos
                self.pacman_direction = (
                    action_index if action_index != 4 else self.pacman_direction
                )
            elif target_type == 1 or target_type == 4:
                pass
            # Pac-Dots
            elif target_type == 2:
                self.pacman_pos = target_pos
                self.pacman_direction = (
                    action_index if action_index != 4 else self.pacman_direction
                )
                if self.maze[approximate_player_pos[0]][approximate_player_pos[1]] == 2:
                    self.maze[approximate_player_pos[0]][approximate_player_pos[1]] = 0
                self.score += PAC_DOTS_POINT
            # Energizer
            elif target_type == 3:
                self.pacman_pos = target_pos
                self.pacman_direction = (
                    action_index if action_index != 4 else self.pacman_direction
                )
                if self.maze[approximate_player_pos[0]][approximate_player_pos[1]] == 3:
                    self.maze[approximate_player_pos[0]][approximate_player_pos[1]] = 0
                self.score += ENERGIZER_POINT
            elif target_type == 6:
                # ghost collision
                self.respawn()

        # Ghost position update
        # TODO: use A* to find the next ghost action
        ghost_action = np.random.randint(4, size=GHOST_COUNT)
        for i in range(GHOST_COUNT):
            ghost_target_pos = self.direction_to_target_position(
                ghost_action[i], self.ghost_pos[i], self.ghost_direction[i]
            )
            ghost_target_type = self.evaluate_target_type(
                ghost_target_pos, ghost_action[i], self.ghost_direction[i], 6
            )

            # Pacman reach a empty cell
            if (
                    ghost_target_type == 0
                    or ghost_target_type == 2
                    or ghost_target_type == 3
                    or ghost_target_type == 6
            ):
                self.ghost_pos[i] = ghost_target_pos
                self.ghost_direction[i] = (
                    ghost_action[i] if ghost_action[i] != 4 else self.ghost_direction[i]
                )
            # Wall collision
            elif ghost_target_type == 1 or ghost_target_type == 4:
                pass
            # Pacman collision
            elif ghost_target_type == 5:
                self.respawn()

        if self.remaining_actions == 0 or self.heart == 0:
            self.game_over = True

    def respawn(self):
        self.heart -= 1
        self.pacman_pos = PACMAN_SPAWN_POINT
        self.pacman_direction = PACMAN_SPAWN_DIRECTION

    def direction_to_target_position(
            self, action_index: int, player_pos: np.ndarray, player_direction: int
    ) -> np.ndarray:
        if action_index == 4:
            action_index = player_direction
        return np.add(player_pos, self.action_vector[action_index])

    # Evaluate the target pos regarding the current pos with the sprite size
    # target_type if center and sprite_pos in cell
    # pacman_type else
    def evaluate_target_type(
            self,
            target_pos: np.ndarray,
            action_index: int,
            player_direction: int,
            player_type: int,
    ) -> int:
        sprite_vector = np.array([0.0, 0.0])
        # Case Noop action: keep the same direction
        if action_index == 4:
            action_index = player_direction

        if action_index == 0:
            sprite_vector = [-SPRITE_SIZE, 0]
        elif action_index == 1:
            sprite_vector = [SPRITE_SIZE, 0]
        elif action_index == 2:
            sprite_vector = [0, -SPRITE_SIZE]
        elif action_index == 3:
            sprite_vector = [0, SPRITE_SIZE]

        center_target_pos = target_pos
        center_target_pos_up = np.ceil(center_target_pos)
        center_target_pos_down = np.floor(center_target_pos)
        sprite_target_pos = np.add(target_pos, sprite_vector)
        sprite_target_pos_up = np.ceil(sprite_target_pos)
        sprite_target_pos_down = np.floor(sprite_target_pos)

        p_approximate_pos = np.floor(self.pacman_pos).astype(int)
        for g_pos in self.ghost_pos:
            if p_approximate_pos[0] == g_pos[0] and p_approximate_pos[1] == g_pos[1]:
                # Ghost collision
                return 6 if player_type == 5 else 5

        # Pacman reach a new cell
        if np.array_equal(
                center_target_pos_up, sprite_target_pos_up
        ) and np.array_equal(center_target_pos_down, sprite_target_pos_down):
            approximate_pos = np.floor(center_target_pos).astype(int)
            if self.maze[approximate_pos[0]][approximate_pos[1]] != 1:
                return self.maze[approximate_pos[0]][approximate_pos[1]]
            else:
                return 1

        # Player position in the maze didn't change
        return player_type

    def get_ghost_action(self):
        pass

    def get_scores(self) -> np.ndarray:
        return self.score

    def get_available_actions(self, player_index: int) -> List[int]:
        return self.available_actions

    def __str__(self):
        str_acc = f"Game Over: {self.game_over}{os.linesep}"
        str_acc += f"Remaining actions: {self.remaining_actions}{os.linesep}"
        str_acc += f"Scores: {self.score[0]} | "
        str_acc += f"Heart: {self.heart}{os.linesep}"
        str_acc += f"Pacman position: {self.pacman_pos}{os.linesep}"
        str_acc += f"Pacman direction: {self.pacman_direction}{os.linesep}"

        for i, line in enumerate(self.maze):
            for j, cell_type in enumerate(line):
                g = False
                p = False
                p_approximate_pos = np.floor(self.pacman_pos).astype(int)
                if i == p_approximate_pos[0] and j == p_approximate_pos[1] and cell_type != 1:
                    str_acc += "ᗧ"
                    p = True
                if not p:
                    for ghost in self.ghost_pos:
                        g_approximate_pos = np.floor(ghost).astype(int)
                        # Display one ghost on the same cell
                        if (
                                i == g_approximate_pos[0]
                                and j == g_approximate_pos[1]
                                and cell_type != 1
                                and not g
                        ):
                            str_acc += "ᗣ"
                            g = True
                if p or g:
                    pass
                elif cell_type == 0:
                    str_acc += " "
                elif cell_type == 1:
                    str_acc += "⯐"
                elif cell_type == 2:
                    str_acc += "•"
                elif cell_type == 3:
                    str_acc += "○"
                elif cell_type == 4:
                    str_acc += "-"

            str_acc += f"{os.linesep}"

        return str_acc

    def get_unique_id(self) -> str:
        return hashlib.sha1(self.get_vectorized_state()).hexdigest()

    def get_max_state_count(self) -> int:
        return self.maze.shape[0] * self.maze.shape[1]

    def get_action_space_size(self) -> int:
        return len(self.available_actions)

    def get_vectorized_state(self, mode: str = None) -> np.ndarray:
        return np.insert(self.ghost_pos, 0, self.pacman_pos)

    def render(self):
        if not self.screen and not self.font:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.font = pygame.font.SysFont("Times New Roman", 20)
            pygame.display.set_caption("Pac-Man")
        self.screen.fill(BLACK)
        if not self.game_over:
            score_text = self.font.render(f"{self.score[0]}", True, WHITE)
            score_text_y = score_text.get_rect().height
            self.screen.blit(
                score_text, (4, (HEIGHT - score_text_y))
            )

            heart_text = self.font.render(f"♥{self.heart}", True, WHITE)
            heart_text_x = heart_text.get_rect().width
            heart_text_y = score_text.get_rect().height
            self.screen.blit(
                heart_text, ((WIDTH - (heart_text_x + 6)), (HEIGHT - heart_text_y))
            )

            arrow = "↑" if self.pacman_direction == 0 else \
                    "↓" if self.pacman_direction == 1 else \
                    "←" if self.pacman_direction == 2 else \
                    "→" if self.pacman_direction == 3 else \
                    "↮"
            action_text = self.font.render(f"{arrow}", True, WHITE)
            action_text_x = action_text.get_rect().width
            action_text_y = action_text.get_rect().height
            self.screen.blit(
                action_text, (((WIDTH/2) - (action_text_x / 2)), (HEIGHT - action_text_y))
            )

            for i, line in enumerate(self.maze):
                for j, cell_type in enumerate(line):
                    if cell_type == 1:
                        pygame.draw.rect(
                            self.screen,
                            BLUE,
                            (
                                j * SQUARE_SIZE,
                                i * SQUARE_SIZE + SQUARE_SIZE,
                                SQUARE_SIZE,
                                SQUARE_SIZE,
                            ),
                        )
                    elif cell_type == 2:
                        pygame.draw.circle(
                            self.screen,
                            TUMBLEWEED,
                            (
                                int(j * SQUARE_SIZE + SQUARE_SIZE / 2),
                                int(i * SQUARE_SIZE + SQUARE_SIZE / 2) + SQUARE_SIZE,
                            ),
                            2
                        )
                    elif cell_type == 3:
                        pygame.draw.circle(
                            self.screen,
                            TUMBLEWEED,
                            (
                                int(j * SQUARE_SIZE + SQUARE_SIZE / 2),
                                int(i * SQUARE_SIZE + SQUARE_SIZE / 2) + SQUARE_SIZE,
                            ),
                            6
                        )
                    elif cell_type == 4:
                        pygame.draw.rect(
                            self.screen,
                            PINK,
                            (
                                j * SQUARE_SIZE,
                                i * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE/2,
                                SQUARE_SIZE,
                                SQUARE_SIZE/2,
                            ),
                        )

                    for k, ghost in enumerate(self.ghost_pos):
                        g_approximate_pos = np.floor(ghost).astype(int)
                        pygame.draw.rect(
                            self.screen,
                            GHOST_COLOR[k],
                            (
                                g_approximate_pos[1] * SQUARE_SIZE,
                                g_approximate_pos[0] * SQUARE_SIZE + SQUARE_SIZE,
                                SQUARE_SIZE,
                                SQUARE_SIZE,
                            ),
                        )
                    p_approximate_pos = np.floor(self.pacman_pos).astype(int)
                    pygame.draw.circle(
                        self.screen,
                        YELLOW,
                        (
                            int(p_approximate_pos[1] * SQUARE_SIZE + SQUARE_SIZE / 2),
                            int(p_approximate_pos[0] * SQUARE_SIZE + SQUARE_SIZE / 2) + SQUARE_SIZE,
                        ),
                        8
                    )

        pygame.display.update()
        pygame.time.delay(1)
