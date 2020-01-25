from drl_gym.agents import RandomRolloutAgent
from drl_gym.environments.tictactoe import TicTacToeGameState
from drl_gym.runners import run_for_n_games_and_print_stats

if __name__ == "__main__":
    gs = TicTacToeGameState()
    agent0 = RandomRolloutAgent(100, False)
    agent1 = RandomRolloutAgent(100, False)

    run_for_n_games_and_print_stats([agent0, agent1], gs, 100)
