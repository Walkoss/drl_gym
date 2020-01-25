import time
from random import shuffle
from typing import List
import numpy as np
import tqdm

from drl_gym.contracts import Agent, GameState
from drl_gym.utils import write_experiment_row


def run_step(agents: List[Agent], gs: GameState):
    assert not gs.is_game_over()
    active_player_index = gs.get_active_player()

    old_scores = gs.get_scores().copy()
    action = agents[active_player_index].act(gs)
    gs.step(active_player_index, action)
    new_scores = gs.get_scores()
    rewards = new_scores - old_scores
    for i, agent in enumerate(agents):
        agent.observe(rewards[i], gs.is_game_over(), i)


def run_to_the_end(agents: List[Agent], gs: GameState, render: bool = False):
    while not gs.is_game_over():
        if render:
            gs.render()
        run_step(agents, gs)

    if render:
        gs.render()


def run_for_n_games_and_return_stats(
    agents: List[Agent],
    gs: GameState,
    games_count: int,
    shuffle_players: bool = False,
    render: bool = False,
    writer=None,
    show_progress: bool = False,
    progress_desc: str = None,
) -> (np.ndarray, np.ndarray, float):
    total_scores = np.zeros_like(gs.get_scores())
    total_times = 0
    agents_order = np.arange(len(agents))

    agents_copy = agents
    if shuffle_players:
        agents_copy = agents.copy()
    iterable = (
        tqdm.tqdm(range(games_count), progress_desc)
        if show_progress
        else range(games_count)
    )
    for game in iterable:
        game = game + 1
        gs_copy = gs.clone()
        if shuffle_players:
            agents_copy = agents.copy()
            shuffle(agents_order)
            for i in agents_order:
                agents_copy[i] = agents[agents_order[i]]
        start = time.time()
        run_to_the_end(agents_copy, gs_copy, render=render)
        total_times += time.time() - start
        total_scores += gs_copy.get_scores()[agents_order]
        if writer:
            mean_scores = total_scores / game
            mean_time_per_game = total_times / game
            write_experiment_row(writer, game, mean_scores, mean_time_per_game)

    return total_scores, total_scores / games_count, total_times / games_count


def run_for_n_games_and_print_stats(
    agents: List[Agent],
    gs: GameState,
    games_count: int,
    shuffle_players: bool = False,
    render: bool = False,
):
    total_scores, mean_scores, mean_times = run_for_n_games_and_return_stats(
        agents, gs, games_count, shuffle_players=shuffle_players, render=render
    )

    print(f"Total Scores : {total_scores}")
    print(f"Mean Scores : {mean_scores}")
    print(f"Mean Times : {mean_times}")


def run_for_n_games_and_return_max(
    agents: List[Agent], gs: GameState, games_count: int, render: bool = False
) -> np.ndarray:
    old_and_new_scores = np.ones((2, len(gs.get_scores()))) * np.NINF

    for _ in range(games_count):
        gs_copy = gs.clone()
        run_to_the_end(agents, gs_copy, render=render)
        new_scores = gs_copy.get_scores()
        old_and_new_scores[1, :] = new_scores
        old_and_new_scores[0, :] = np.max(old_and_new_scores, axis=0)

    return old_and_new_scores[0, :]
