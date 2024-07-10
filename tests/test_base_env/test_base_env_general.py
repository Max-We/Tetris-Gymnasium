import gymnasium as gym
import pytest

from tetris_gymnasium.envs.tetris import Tetris

@pytest.fixture
def tetris_env():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env.reset(seed=42)
    yield env
    env.close()

def test_observation_space_is_correct_after_reset(tetris_env):
    """Test that the observation space keys are correct after resetting the environment."""
    observation, info = tetris_env.reset(seed=42)
    assert tetris_env.observation_space.keys() == observation.keys()
