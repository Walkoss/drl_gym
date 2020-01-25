from drl_gym.agents import RandomRolloutAgent
from drl_gym.environments import GridWorldGameState
from drl_gym.runners import run_to_the_end

if __name__ == "__main__":
    gs = GridWorldGameState()
    agent = RandomRolloutAgent(100000, True)

    print(gs)
    run_to_the_end([agent], gs)
    print(gs)
