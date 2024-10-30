import jax
import jax.numpy as jnp
import pytest

from tetris_gymnasium.envs.tetris_fn import reset, step
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.tetrominoes import TETROMINOES


@pytest.fixture
def env_config():
    return EnvConfig(width=10, height=20, padding=4, queue_size=7)


@pytest.fixture
def initial_state(env_config):
    key = jax.random.PRNGKey(42)
    key, state, obs = reset(TETROMINOES, key, env_config)
    return key, state


def test_line_clear_normal(env_config, initial_state):
    """Test normal line clear with vertical I-tetromino."""
    key, state = initial_state

    # Fill board except the last column
    filled_board = state.board.at[
        env_config.height - 4 : env_config.height,
        env_config.padding : -env_config.padding - 1,
    ].set(1)
    state = state.replace(board=filled_board)

    # Set I-tetromino
    state = state.replace(
        active_tetromino=0,
        rotation=1,  # Vertical orientation
        x=env_config.width
        + env_config.padding
        - 2,  # subtract 1 because of 0-based index
        y=0,
    )

    # Perform hard drop
    key, new_state, new_obs, reward, done, info = step(
        TETROMINOES, key, state, 6, env_config
    )

    # Check that the board has been cleared
    empty_board = state.board.at[
        env_config.height - 4 : env_config.height,
        env_config.padding : -env_config.padding,
    ].set(0)
    assert jnp.all(new_state.board == empty_board)

    # Check that the reward is correct (you may need to adjust this based on your reward system)
    expected_reward = (4**2) * env_config.width  # Assuming same reward formula
    assert reward == expected_reward

    # Check that the game is not over
    assert not done

    # Check that a new tetromino has been spawned and x, y have been reset
    assert new_state.active_tetromino != state.active_tetromino
    assert new_state.y == 0


# Run the tests
if __name__ == "__main__":
    # for debugging, otherwise can't see concrete values (only tracer)
    jax.config.update("jax_disable_jit", True)
    pytest.main([__file__])
