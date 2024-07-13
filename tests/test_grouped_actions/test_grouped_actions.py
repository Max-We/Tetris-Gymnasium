import os

import gymnasium as gym
import numpy as np
import pytest

from tests.helpers.mock import generate_example_board_with_features, convert_to_base_observation
from tetris_gymnasium.wrappers.action import GroupedActions
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


@pytest.fixture
def tetris_env_grouped():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env = GroupedActions(env)
    env.reset(seed=42)
    yield env
    env.close()

@pytest.fixture
def tetris_env_grouped_wrappers():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env = GroupedActions(env, observation_wrappers=[FeatureVectorObservation(env, report_height=True, report_max_height=True, report_holes=True, report_bumpiness=True)])
    env.reset(seed=42)
    yield env
    env.close()

@pytest.fixture
def expected_result_i_placement():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV file
    csv_file_path = os.path.join(script_dir, "data", "expected_result_i_placement.csv")
    return np.genfromtxt(csv_file_path, delimiter=",").astype(np.uint8)

def test_observation_space_is_correct_after_reset(tetris_env_grouped):
    """Test that the observation space is correct after resetting the environment."""
    assert tetris_env_grouped.observation_space.shape == (40, 24, 18)

def test_observation_space_is_correct_after_reset_with_other_wrappers(tetris_env_grouped_wrappers):
    """Test that the observation space is correct after resetting the environment."""
    assert tetris_env_grouped_wrappers.observation_space.shape == (40, 10)

def test_action_mapping_is_correct(tetris_env_grouped, vertical_i_tetromino, expected_result_i_placement):
    """Test that the observation space is correct after resetting the environment."""
    example_board, _, _, _, _ = generate_example_board_with_features(
        tetris_env_grouped
    )
    tetris_env_grouped.unwrapped.board = example_board
    tetris_env_grouped.unwrapped.active_tetromino = vertical_i_tetromino

    tetris_env_grouped.step(13) # expected: column (index) 3, rotation 1

    assert np.all(tetris_env_grouped.unwrapped.board == expected_result_i_placement)

def test_observation_index_is_correct(tetris_env_grouped, vertical_i_tetromino, expected_result_i_placement):
    """Test that the observation space is correct after resetting the environment."""
    example_board, _, _, _, _ = generate_example_board_with_features(
        tetris_env_grouped
    )
    base_observation = convert_to_base_observation(example_board)

    # Why need to set board, if observation is passed with information?
    # Clear up impl.
    tetris_env_grouped.unwrapped.board = example_board
    tetris_env_grouped.unwrapped.active_tetromino = vertical_i_tetromino

    observation = tetris_env_grouped.observation(base_observation)
    assert observation.shape == (40, 24, 18)

    assert np.all(observation[13] == expected_result_i_placement) # expected: column (index) 3, rotation 1

def test_legal_action_mask_is_correct(tetris_env_grouped, vertical_i_tetromino):
    """Test that the observation space is correct after resetting the environment."""
    example_board, _, _, _, _ = generate_example_board_with_features(
        tetris_env_grouped
    )
    base_observation = convert_to_base_observation(example_board)

    tetris_env_grouped.unwrapped.board = example_board
    tetris_env_grouped.unwrapped.active_tetromino = vertical_i_tetromino

    # Generate action mask
    _ = tetris_env_grouped.observation(base_observation)

    expected_action_mask = np.ones(40)
    expected_action_mask = expected_action_mask.reshape(4,10, order="F")
    expected_action_mask[:,9] = 0
    expected_action_mask[[1,3],7] = 0
    # the other vertical orientation (2) is also illegal because padding is larger on the left side
    expected_action_mask[[1,2,3],8] = 0
    expected_action_mask = expected_action_mask.reshape(40, order="F")

    assert np.all(tetris_env_grouped.legal_actions_mask.astype(np.uint8) == expected_action_mask.astype(np.uint8)) # expected: column (index) 3, rotation 1

# def test_observation_values_with_wrappers(tetris_env_with_wrappers, o_tetromino):
#     """Test that the observation space is correct after resetting the environment."""
#     example_board, _, _, _, correct_bumpiness = generate_example_board(
#         tetris_env_with_wrappers
#     )
#     tetris_env_with_wrappers.unwrapped.board = example_board
#
#     observation, _, _, _, _ = tetris_env_with_wrappers.step(0) # column 0, rotation 0
#     assert

# def test_rgb_values_are_valid(tetris_env):
#     """Test that the RGB values are valid after taking an action."""
#     observation, _, _, _, _ = tetris_env.step(ActionsMapping.hard_drop)
#     assert observation.min() >= 0
#     assert observation.max() <= 255
#     assert observation.shape == (24, 34, 3)
#     assert observation.dtype == "uint8"
