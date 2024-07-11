import gymnasium as gym
import numpy as np
import pytest

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_gymnasium.wrappers.observation import (
    FeatureVectorObservation,
    RgbObservation,
)


@pytest.fixture
def tetris_env_height():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi", gravity=False)
    env = FeatureVectorObservation(
        env,
        report_height=True,
        report_max_height=False,
        report_holes=False,
        report_bumpiness=False,
    )
    env.reset(seed=42)
    yield env
    env.close()


@pytest.fixture
def tetris_env_max_height():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi", gravity=False)
    env = FeatureVectorObservation(
        env,
        report_height=False,
        report_max_height=True,
        report_holes=False,
        report_bumpiness=False,
    )
    env.reset(seed=42)
    yield env
    env.close()


@pytest.fixture
def tetris_env_holes():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi", gravity=False)
    env = FeatureVectorObservation(
        env,
        report_height=False,
        report_max_height=False,
        report_holes=True,
        report_bumpiness=False,
    )
    env.reset(seed=42)
    yield env
    env.close()


@pytest.fixture
def tetris_env_bumpiness():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi", gravity=False)
    env = FeatureVectorObservation(
        env,
        report_height=False,
        report_max_height=False,
        report_holes=False,
        report_bumpiness=True,
    )
    env.reset(seed=42)
    yield env
    env.close()


def generate_example_board(env):
    """
    Generate an example Tetris board with specific characteristics for testing.

    Args:
        env: The Tetris environment.

    Returns:
        A tuple containing:
        - board: The generated board.
        - height: Array of column heights.
        - max_height: Maximum height of the board.
        - holes: Number of holes in the board.
        - bumpiness: Measure of the board's bumpiness.
    """
    board = np.copy(env.unwrapped.board)
    # fill half the rows (except the last column to prevent line clear)
    target_height = env.unwrapped.height // 2
    board[
        target_height : env.unwrapped.height,
        env.unwrapped.padding : -(env.unwrapped.padding + 1),
    ] = 2
    # add some bumpiness
    board[target_height - 1, env.unwrapped.padding + 1] = 2
    board[target_height - 1, env.unwrapped.padding + 4] = 2
    board[target_height - 1, env.unwrapped.padding + 5] = 2
    # add some holes
    board[target_height + 2, env.unwrapped.padding + 2] = 0
    board[target_height + 4, env.unwrapped.padding + 3] = 0
    board[target_height + 6, env.unwrapped.padding + 6] = 0

    max_height = target_height + 1
    height = [target_height] * 10
    height[1], height[4], height[5], height[9] = max_height, max_height, max_height, 0
    holes = 3
    bumpiness = 14

    return (
        board,
        np.array(height, dtype=np.uint8),
        np.array([max_height], dtype=np.uint8),
        np.array([holes], dtype=np.uint8),
        np.array([bumpiness], dtype=np.uint8),
    )


# Tests for observation shapes


def test_height_observation_shape(tetris_env_height):
    """Test if the height observation shape is correct."""
    assert tetris_env_height.observation_space.shape == (
        tetris_env_height.unwrapped.width,
    )
    observation, _ = tetris_env_height.reset(seed=42)
    assert observation.shape == (tetris_env_height.unwrapped.width,)


def test_max_height_observation_shape(tetris_env_max_height):
    """Test if the max height observation shape is correct."""
    assert tetris_env_max_height.observation_space.shape == (1,)
    observation, _ = tetris_env_max_height.reset(seed=42)
    assert observation.shape == (1,)


def test_holes_observation_shape(tetris_env_holes):
    """Test if the holes observation shape is correct."""
    assert tetris_env_holes.observation_space.shape == (1,)
    observation, _ = tetris_env_holes.reset(seed=42)
    assert observation.shape == (1,)


def test_bumpiness_observation_shape(tetris_env_bumpiness):
    """Test if the bumpiness observation shape is correct."""
    assert tetris_env_bumpiness.observation_space.shape == (1,)
    observation, _ = tetris_env_bumpiness.reset(seed=42)
    assert observation.shape == (1,)


# Tests for observation values


def test_height_observation_values(tetris_env_height):
    """Test if the height observation values are correct."""
    example_board, correct_height, _, _, _ = generate_example_board(tetris_env_height)
    tetris_env_height.unwrapped.board = example_board
    observation, _, _, _, _ = tetris_env_height.step(ActionsMapping.no_op)
    assert np.all(observation == correct_height)


def test_max_height_observation_values(tetris_env_max_height):
    """Test if the max height observation values are correct."""
    example_board, _, correct_max_height, _, _ = generate_example_board(
        tetris_env_max_height
    )
    tetris_env_max_height.unwrapped.board = example_board
    observation, _, _, _, _ = tetris_env_max_height.step(ActionsMapping.no_op)
    assert np.all(observation == correct_max_height)


def test_holes_observation_values(tetris_env_holes):
    """Test if the holes observation values are correct."""
    example_board, _, _, correct_holes, _ = generate_example_board(tetris_env_holes)
    tetris_env_holes.unwrapped.board = example_board
    observation, _, _, _, _ = tetris_env_holes.step(ActionsMapping.no_op)
    assert np.all(observation == correct_holes)


def test_bumpiness_observation_values(tetris_env_bumpiness):
    """Test if the bumpiness observation values are correct."""
    example_board, _, _, _, correct_bumpiness = generate_example_board(
        tetris_env_bumpiness
    )
    tetris_env_bumpiness.unwrapped.board = example_board
    observation, _, _, _, _ = tetris_env_bumpiness.step(ActionsMapping.no_op)
    assert np.all(observation == correct_bumpiness)
