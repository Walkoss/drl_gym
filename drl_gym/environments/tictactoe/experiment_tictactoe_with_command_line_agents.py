from drl_gym.agents import CommandLineAgent
from drl_gym.environments.tictactoe import TicTacToeGameState
from drl_gym.runners import run_to_the_end

if __name__ == "__main__":
    gs = TicTacToeGameState()
    agent = CommandLineAgent()

    print(gs)
    run_to_the_end([agent, agent], gs)
    print(gs)
