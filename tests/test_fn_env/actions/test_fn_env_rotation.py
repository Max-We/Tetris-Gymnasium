import jax
import pytest

from tetris_gymnasium.envs.tetris_fn import reset, step
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.tetrominoes import TETROMINOES


@pytest.fixture
def env_config():
    return EnvConfig(
        width=10, height=20, padding=4, queue_size=7, gravity_enabled=False
    )


@pytest.fixture
def initial_state(env_config):
    key = jax.random.PRNGKey(42)
    key, state, obs = reset(TETROMINOES, key, env_config)
    return key, state


def test_rotate_clockwise(env_config, initial_state):
    """Test rotating a tetromino clockwise in a regular scenario."""
    key, state = initial_state
    initial_rotation = state.rotation
    key, new_state, new_obs, reward, done, info = step(
        TETROMINOES, key, state, 4, env_config
    )
    assert new_state.rotation == (initial_rotation + 1) % 4


def test_rotate_counterclockwise(env_config, initial_state):
    """Test rotating a tetromino counter-clockwise in a regular scenario."""
    key, state = initial_state
    initial_rotation = state.rotation
    key, new_state, new_obs, reward, done, info = step(
        TETROMINOES, key, state, 3, env_config
    )
    assert new_state.rotation == (initial_rotation - 1) % 4


def test_rotate_clockwise_blocked(env_config, initial_state):
    """Test rotating a tetromino clockwise when blocked by another tetromino."""
    key, state = initial_state
    # Place blocks to potentially block rotation
    state = state.replace(board=state.board.at[state.y + 1, state.x + 2].set(1))
    initial_rotation = state.rotation
    key, new_state, new_obs, reward, done, info = step(
        TETROMINOES, key, state, 3, env_config
    )
    # If rotation is blocked, the rotation should remain the same
    assert new_state.rotation == initial_rotation


def test_rotate_counterclockwise_blocked(env_config, initial_state):
    """Test rotating a tetromino counter-clockwise when blocked by another tetromino."""
    key, state = initial_state
    # Place blocks to potentially block rotation
    state = state.replace(board=state.board.at[state.y + 1, state.x].set(1))
    initial_rotation = state.rotation
    key, new_state, new_obs, reward, done, info = step(
        TETROMINOES, key, state, 4, env_config
    )
    # If rotation is blocked, the rotation should remain the same
    assert new_state.rotation == initial_rotation


# Run the tests
if __name__ == "__main__":
    # for debugging, otherwise can't see concrete values (only tracer)
    jax.config.update("jax_disable_jit", True)
    pytest.main([__file__])
