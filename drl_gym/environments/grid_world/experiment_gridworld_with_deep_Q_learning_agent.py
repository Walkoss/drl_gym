from drl_gym.agents import DeepQLearningAgent
from drl_gym.environments import GridWorldGameState
from drl_gym.runners import run_for_n_games_and_print_stats, run_step

if __name__ == "__main__":
    gs = GridWorldGameState()
    agent = DeepQLearningAgent(action_space_size=gs.get_action_space_size())

    for i in range(500):
        run_for_n_games_and_print_stats([agent], gs, 100)

    agent.epsilon = -1.0
    run_for_n_games_and_print_stats([agent], gs, 100)

    gs = gs.clone()
    while not gs.is_game_over():
        run_step([agent], gs)
        print(gs)
