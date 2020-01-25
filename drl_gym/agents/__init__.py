from .random_agent import RandomAgent
from .command_line_agent import CommandLineAgent
from .tabular_q_learning_agent import TabQLearningAgent
from .deep_q_learning_agent import DeepQLearningAgent
from .tabular_like_deep_q_learning_agent import TabularLikeDeepQLearningAgent
from .random_rollout_agent import RandomRolloutAgent
from .ppo_agent import PPOAgent
from .MO_MCTS_agent import MOMCTSAgent
from .half_alphazero_agent import HalfAlphaZeroAgent
from .MO_MCTS_expert_FNN_apprentice_agent import ExpertApprenticeAgent

available_agents = [
    "Random",
    "CommandLine",
    "TabQLearning",
    "DeepQLearning",
    "TabularLikeDeepQLearning",
    "RandomRollout",
    "PPO",
    "MOMCTS",
    "HalfAlphaZero",
    "ExpertApprentice",
]
