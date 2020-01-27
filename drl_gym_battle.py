import argparse
import ast
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from typing import List

import drl_gym

from drl_gym.environments import available_game_states
from drl_gym.agents import available_agents
from drl_gym.contracts import GameState, Agent
from drl_gym.runners import run_for_n_games_and_return_stats
from drl_gym.utils import get_experiment_csv_writer


DEFAULT_GAMES_COUNT = 100


def check_positive(value_to_check):
    casted_value = int(value_to_check)
    if casted_value <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value_to_check
        )
    return casted_value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("game_state", type=str)
    parser.add_argument("agents", nargs="+", type=str)
    parser.add_argument(
        "-c", "--games-count", type=check_positive, default=DEFAULT_GAMES_COUNT
    )
    parser.add_argument("--agent-params", type=ast.literal_eval, action="append")
    parser.add_argument(
        "-r", "--render", nargs="?", const=True, default=False,
    )
    parser.add_argument("--game-state-params", type=ast.literal_eval)
    parser.add_argument("--load-models", type=str, action="append")
    return parser.parse_args()


def agent_params_to_str(agent_param: dict):
    if agent_param:
        return "_".join(f"{key}{value}" for key, value in agent_param.items())
    return "default_params"


if __name__ == "__main__":
    args = parse_args()

    # Check game state argument
    assert (
        args.game_state in available_game_states
    ), f"Incorrect game state '{args.game_state}', choose from {available_game_states}"
    game_state_params = {}
    if args.game_state_params:
        game_state_params = args.game_state_params
    gs: GameState = getattr(drl_gym.environments, f"{args.game_state}GameState")(
        **game_state_params
    )

    # Check agents argument
    agent_params = [dict()] * gs.player_count()
    if args.agent_params:
        for i, value in enumerate(args.agent_params):
            agent_params[i] = value

    assert gs.player_count() == len(
        args.agents
    ), f"You must select {gs.player_count()} agents for {args.game_state} environment"
    agents: List[Agent] = []
    for i, agent in enumerate(args.agents):
        assert (
            agent in available_agents
        ), f"Incorrect agent '{agent}', choose from {available_agents}"
        agents.append(getattr(drl_gym.agents, f"{agent}Agent")(**agent_params[i]))

    if args.load_models:
        for i, param in enumerate(args.load_models):
            agents[i].load_model(param)

    # Create log file
    log_filename = f"{args.game_state}"
    for i, agent in enumerate(args.agents):
        log_filename += f"_{agent}"
        if agent_params[i]:
            log_filename += f"_{agent_params_to_str(agent_params[i])}"

    f = open(f"logs/{log_filename}.csv", "w", newline="")
    writer = get_experiment_csv_writer(f, gs.player_count())

    # Run battle
    try:
        run_for_n_games_and_return_stats(
            agents,
            gs,
            args.games_count,
            writer=writer,
            show_progress=True,
            progress_desc=f"{'_'.join(args.agents)} on {args.games_count} games",
            render=args.render,
        )
    finally:
        for i, agent in enumerate(agents):
            agent.save_model(
                f"models/{args.game_state}_{agent.__class__.__name__}{i}_{agent_params_to_str(agent_params[i])}"
            )
        f.close()
