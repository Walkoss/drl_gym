from drl_gym.agents import TabQLearningAgent, CommandLineAgent
from drl_gym.environments.tictactoe import TicTacToeGameState
from drl_gym.runners import run_for_n_games_and_print_stats, run_step

if __name__ == "__main__":
    gs = TicTacToeGameState()
    agent0 = TabQLearningAgent()
    agent1 = TabQLearningAgent()
    agent0.alpha = 0.1
    agent0.epsilon = 0.005
    agent1.alpha = 0.1
    agent1.epsilon = 0.005

    for _ in range(100):
        run_for_n_games_and_print_stats([agent0, agent1], gs, 5000)

    agent0.epsilon = -1.0
    agent1.epsilon = -1.0
    run_for_n_games_and_print_stats([agent0, agent1], gs, 100)

    gs_clone = gs.clone()
    while not gs_clone.is_game_over():
        run_step([agent0, CommandLineAgent()], gs_clone)
        print(gs_clone)

    gs_clone = gs.clone()
    while not gs_clone.is_game_over():
        run_step([CommandLineAgent(), agent1], gs_clone)
        print(gs_clone)
