import pygame

from drl_gym.contracts import Agent, GameState


class MouseAgent(Agent):
    def act(self, gs: GameState) -> int:
        available_actions = gs.get_available_actions(gs.get_active_player())
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    valid_action = gs.get_valid_action_from_mouse_pos(
                        pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]
                    )
                    if valid_action in available_actions:
                        return valid_action
                    else:
                        print(f"Incorrect action")

    def observe(self, r: float, t: bool, player_index: int):
        pass

    def save_model(self, filename: str):
        pass

    def load_model(self, filename: str):
        pass
