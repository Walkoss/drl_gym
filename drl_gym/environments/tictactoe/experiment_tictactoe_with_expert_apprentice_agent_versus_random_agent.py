from drl_gym.agents import RandomAgent, ExpertApprenticeAgent
from drl_gym.environments.tictactoe import TicTacToeGameState
from drl_gym.runners import run_for_n_games_and_print_stats

if __name__ == "__main__":

    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    gs = TicTacToeGameState()
    agent0 = ExpertApprenticeAgent(100, gs.get_action_space_size())
    agent1 = RandomAgent()

    for _ in range(1000):
        run_for_n_games_and_print_stats([agent0, agent1], gs, 1000)
