import jax
import jax.numpy as jnp
import pytest

from tetris_gymnasium.envs.tetris_fn import reset, step
from tetris_gymnasium.functional.core import EnvConfig, State
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

@pytest.fixture
def env_config():
    return EnvConfig(width=10, height=20, padding=4, queue_size=7)

@pytest.fixture
def initial_state(env_config):
    key = jax.random.PRNGKey(42)
    key, state = reset(TETROMINOES, key, env_config)
    return key, state

def test_game_over_stack_too_high(env_config, initial_state):
    """Test game over condition when stack is too high after hard drop."""
    key, state = initial_state

    # Fill the board up to the second row from the top
    filled_board = state.board.at[
        2: env_config.height,
        env_config.padding+1: -env_config.padding
    ].set(1)

    # Set O-tetromino (assuming it's the second in the TETROMINOES list)
    state = state.replace(
        board=filled_board,
        active_tetromino=1,  # O-tetromino
        x=env_config.width // 2 + env_config.padding - 1,  # Center the tetromino
        y=0
    )

    # Perform hard drop
    key, new_state, reward, done, info = step(TETROMINOES, key, state, 6, env_config)

    # Check that the game is over
    assert done

    # Check that the reward is correct
    expected_reward = 0
    assert reward == expected_reward

# def test_game_over_invalid_position(env_config, initial_state):
#     """Test game over condition when active tetromino is inside other tetrominos or the wall."""
#     key, state = initial_state
#
#     # Fill the entire board except the padding
#     filled_board = state.board.at[
#         0: env_config.height,
#         env_config.padding: -env_config.padding
#     ].set(1)
#
#     # Set O-tetromino (assuming it's the second in the TETROMINOES list)
#     state = state.replace(
#         board=filled_board,
#         active_tetromino=1,  # O-tetromino
#         x=env_config.width // 2 + env_config.padding - 1,  # Center the tetromino
#         y=0
#     )
#
#     # Perform hard drop
#     key, new_state, reward, done, info = step(TETROMINOES, key, state, 6, env_config)
#
#     # Check that the game is over
#     assert done
#
#     # Check that the reward is correct
#     expected_reward = 0
#     assert reward == expected_reward

# Run the tests
if __name__ == "__main__":
    # for debugging, otherwise cant see concrete values (only tracer)
    jax.config.update("jax_disable_jit", True)
    pytest.main([__file__])