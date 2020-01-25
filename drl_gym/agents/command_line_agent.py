from drl_gym.contracts import Agent, GameState


class CommandLineAgent(Agent):
    def act(self, gs: GameState) -> int:
        available_actions = gs.get_available_actions(gs.get_active_player())
        print(f"Choose action index from : {available_actions}")

        while True:
            try:
                action_candidate = int(input())
                if action_candidate in available_actions:
                    break
            except Exception as _:
                pass
            print(f"Action not valid, please try again !")
        return action_candidate

    def observe(self, r: float, t: bool, player_index: int):
        pass

    def save_model(self, name: str):
        pass
