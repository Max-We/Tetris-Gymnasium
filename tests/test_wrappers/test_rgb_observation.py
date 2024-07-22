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
