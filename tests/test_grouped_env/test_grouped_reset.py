import gymnasium as gym
import pytest

from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations


@pytest.fixture
def tetris_env_grouped_fresh():
    """Fixture for a fresh grouped environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env = GroupedActionsObservations(env)
    yield env
    env.close()


def test_reset_returns_valid_observation_shape(tetris_env_grouped_fresh):
    """Test that reset returns observation with shape (40, h_padded, w_padded)."""
    obs, info = tetris_env_grouped_fresh.reset(seed=42)

    h = tetris_env_grouped_fresh.unwrapped.height_padded
    w = tetris_env_grouped_fresh.unwrapped.width_padded
    expected_shape = (40, h, w)
    assert obs.shape == expected_shape


def test_reset_returns_action_mask_in_info(tetris_env_grouped_fresh):
    """Test that info dict from reset has an action_mask key."""
    obs, info = tetris_env_grouped_fresh.reset(seed=42)

    assert "action_mask" in info


def test_reset_action_mask_has_correct_shape(tetris_env_grouped_fresh):
    """Test that the action mask has shape (40,)."""
    obs, info = tetris_env_grouped_fresh.reset(seed=42)

    assert info["action_mask"].shape == (40,)
