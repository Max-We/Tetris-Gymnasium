import gymnasium as gym
import pytest

from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_gymnasium.wrappers.observation import RgbObservation


@pytest.fixture
def tetris_env_rgb():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env = RgbObservation(env)
    env.reset(seed=42)
    yield env
    env.close()


def test_observation_space_is_correct_after_reset(tetris_env_rgb):
    """Test that the observation space is correct after resetting the environment."""
    assert tetris_env_rgb.observation_space.shape == (24, 34, 3)


def test_rgb_values_are_valid(tetris_env_rgb):
    """Test that the RGB values are valid after taking an action."""
    observation, _, _, _, _ = tetris_env_rgb.step(ActionsMapping.hard_drop)
    assert observation.min() >= 0
    assert observation.max() <= 255
    assert observation.shape == (24, 34, 3)
    assert observation.dtype == "uint8"


def test_rgb_values_within_valid_range(tetris_env_rgb):
    """Test that all RGB pixel values are in [0, 255]."""
    observation, _ = tetris_env_rgb.reset(seed=42)
    assert observation.min() >= 0
    assert observation.max() <= 255


def test_empty_board_has_expected_colors(tetris_env_rgb):
    """Test that empty cells on the board are black (0,0,0)."""
    observation, _ = tetris_env_rgb.reset(seed=42)
    # The top-left area of the board (inside padding) should be the playfield
    # Empty cells should be black [0, 0, 0]
    padding = tetris_env_rgb.unwrapped.padding
    # Check a cell that should be empty (top of the playfield, away from the piece)
    # Row 0, column at padding should be within the playfield
    empty_pixel = observation[0, padding, :]
    assert list(empty_pixel) == [0, 0, 0], "Empty cells should be black"
