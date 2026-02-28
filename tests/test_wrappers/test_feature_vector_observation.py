import gymnasium as gym
import numpy as np
import pytest

from tests.helpers.mock import generate_example_board_with_features
from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation


@pytest.fixture
def tetris_env_all_features():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi", gravity=False)
    env = FeatureVectorObservation(env)
    env.reset(seed=42)
    yield env
    env.close()


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


def test_all_features_observation_shape(tetris_env_all_features):
    """Test if the bumpiness observation shape is correct."""
    assert tetris_env_all_features.observation_space.shape == (13,)
    observation, _ = tetris_env_all_features.reset(seed=42)
    assert observation.shape == (13,)


# Tests for observation values


def test_height_observation_values(tetris_env_height):
    """Test if the height observation values are correct."""
    example_board, correct_height, _, _, _ = generate_example_board_with_features(
        tetris_env_height
    )
    tetris_env_height.unwrapped.board = example_board
    observation, _, _, _, _ = tetris_env_height.step(ActionsMapping.no_op)
    assert np.all(observation == correct_height)


def test_max_height_observation_values(tetris_env_max_height):
    """Test if the max height observation values are correct."""
    example_board, _, correct_max_height, _, _ = generate_example_board_with_features(
        tetris_env_max_height
    )
    tetris_env_max_height.unwrapped.board = example_board
    observation, _, _, _, _ = tetris_env_max_height.step(ActionsMapping.no_op)
    assert np.all(observation == correct_max_height)


def test_holes_observation_values(tetris_env_holes):
    """Test if the holes observation values are correct."""
    example_board, _, _, correct_holes, _ = generate_example_board_with_features(
        tetris_env_holes
    )
    tetris_env_holes.unwrapped.board = example_board
    observation, _, _, _, _ = tetris_env_holes.step(ActionsMapping.no_op)
    assert np.all(observation == correct_holes)


def test_bumpiness_observation_values(tetris_env_bumpiness):
    """Test if the bumpiness observation values are correct."""
    example_board, _, _, _, correct_bumpiness = generate_example_board_with_features(
        tetris_env_bumpiness
    )
    tetris_env_bumpiness.unwrapped.board = example_board
    observation, _, _, _, _ = tetris_env_bumpiness.step(ActionsMapping.no_op)
    assert np.all(observation == correct_bumpiness)


def test_features_on_empty_board(tetris_env_all_features):
    """Test that all features are 0 on an empty board."""
    tetris_env_all_features.reset(seed=42)
    # Create a clean board (no pieces)
    env = tetris_env_all_features.unwrapped
    env.board = env.create_board()

    observation, _, _, _, _ = tetris_env_all_features.step(ActionsMapping.no_op)

    heights = observation[: env.width]
    max_height = observation[env.width]
    holes = observation[env.width + 1]
    bumpiness = observation[env.width + 2]

    assert np.all(heights == 0), "All heights should be 0 on empty board"
    assert max_height == 0, "Max height should be 0 on empty board"
    assert holes == 0, "Holes should be 0 on empty board"
    assert bumpiness == 0, "Bumpiness should be 0 on empty board"


def test_active_tetromino_masked_out(tetris_env_all_features):
    """Test that the active piece doesn't affect feature calculation."""
    tetris_env_all_features.reset(seed=42)
    env = tetris_env_all_features.unwrapped
    env.board = env.create_board()

    # Get features with piece at top (should be masked out)
    env.y = 0
    observation, _, _, _, _ = tetris_env_all_features.step(ActionsMapping.no_op)

    # Even though there's an active tetromino, features should reflect empty board
    # (active tetromino is masked out in the FeatureVectorObservation wrapper)
    heights = observation[: env.width]
    assert np.all(heights == 0), "Active tetromino should not affect height calculation"
