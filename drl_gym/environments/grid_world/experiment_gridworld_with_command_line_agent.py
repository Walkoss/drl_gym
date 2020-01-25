from drl_gym.agents import CommandLineAgent
from drl_gym.environments import GridWorldGameState
from drl_gym.runners import run_to_the_end

if __name__ == "__main__":
    gs = GridWorldGameState()
    agent = CommandLineAgent()

    print(gs)
    run_to_the_end([agent], gs)
    print(gs)
