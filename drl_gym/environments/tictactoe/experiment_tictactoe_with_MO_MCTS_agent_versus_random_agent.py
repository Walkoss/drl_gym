from drl_gym.agents import MOMCTSAgent, RandomAgent
from drl_gym.environments.tictactoe import TicTacToeGameState
from drl_gym.runners import run_for_n_games_and_print_stats

if __name__ == "__main__":
    gs = TicTacToeGameState()
    agent0 = MOMCTSAgent(2)
    agent1 = RandomAgent()

    for _ in range(1000):
        run_for_n_games_and_print_stats([agent0, agent1], gs, 1000)
