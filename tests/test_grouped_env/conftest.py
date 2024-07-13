import os

import gymnasium as gym
import numpy as np
import pytest

from tests.helpers.mock import generate_example_board_with_features, convert_to_base_observation
from tetris_gymnasium.wrappers.action import GroupedActions
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


@pytest.fixture
def tetris_env_grouped(vertical_i_tetromino):
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env = GroupedActions(env)
    env.reset(seed=42)

    example_board, _, _, _, _ = generate_example_board_with_features(
        env
    )
    env.unwrapped.board = example_board
    env.unwrapped.active_tetromino = vertical_i_tetromino

    _ = env.observation(convert_to_base_observation(example_board)) # to update action mask

    yield env
    env.close()

@pytest.fixture
def tetris_env_grouped_wrappers(vertical_i_tetromino):
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env = GroupedActions(env, observation_wrappers=[FeatureVectorObservation(env, report_height=True, report_max_height=True, report_holes=True, report_bumpiness=True)])

    example_board, _, _, _, _ = generate_example_board_with_features(
        env
    )
    env.unwrapped.board = example_board
    env.unwrapped.active_tetromino = vertical_i_tetromino

    _ = env.observation(convert_to_base_observation(example_board)) # to update action mask

    env.reset(seed=42)
    yield env
    env.close()

@pytest.fixture
def expected_result_i_placement():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV file
    csv_file_path = os.path.join(script_dir, "expected_result_i_placement.csv")
    return np.genfromtxt(csv_file_path, delimiter=",").astype(np.uint8)

@pytest.fixture
def base_observation(tetris_env_grouped):
    example_board, _, _, _, _ = generate_example_board_with_features(
        tetris_env_grouped
    )
    return convert_to_base_observation(example_board)