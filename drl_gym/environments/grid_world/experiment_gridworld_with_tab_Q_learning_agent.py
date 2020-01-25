from drl_gym.agents import TabQLearningAgent
from drl_gym.environments import GridWorldGameState
from drl_gym.runners import run_for_n_games_and_print_stats, run_step

if __name__ == "__main__":
    gs = GridWorldGameState()
    agent = TabQLearningAgent()

    for _ in range(500):
        run_for_n_games_and_print_stats([agent], gs, 100)

    agent.epsilon = -1.0
    run_for_n_games_and_print_stats([agent], gs, 100)

    gs = gs.clone()
    while not gs.is_game_over():
        run_step([agent], gs)
        print(gs)
