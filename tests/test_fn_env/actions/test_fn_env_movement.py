import jax
import jax.numpy as jnp
import pytest

from tetris_gymnasium.envs.tetris_fn import reset, step
from tetris_gymnasium.functional.core import EnvConfig, State
from tetris_gymnasium.functional.queue import (
    bag_queue_get_next_element,
    create_bag_queue,
)
from tetris_gymnasium.functional.tetrominoes import TETROMINOES


@pytest.fixture
def env_config():
    return EnvConfig(width=10, height=20, padding=4, queue_size=7)


@pytest.fixture
def initial_state(env_config):
    key = jax.random.PRNGKey(42)
    key, state = reset(TETROMINOES, key, env_config)
    return key, state


def test_reset(env_config, initial_state):
    key, state = initial_state
    assert isinstance(state, State)
    assert state.board.shape == (
        env_config.height + env_config.padding,
        env_config.width + 2 * env_config.padding,
    )
    assert 0 <= state.active_tetromino < env_config.queue_size
    assert state.rotation == 0
    assert env_config.padding <= state.x < env_config.width + env_config.padding
    assert state.y == 0
    assert state.queue.shape == (env_config.queue_size,)
    assert state.queue_index == 1
    assert not state.game_over
    assert state.score == 0


def test_move_right(env_config, initial_state):
    key, state = initial_state
    initial_x = state.x
    key, new_state, reward, done, info = step(TETROMINOES, key, state, 1, env_config)
    assert new_state.x == initial_x + 1
    assert new_state.y == state.y + 1  # Due to gravity


def test_move_left(env_config, initial_state):
    key, state = initial_state
    initial_x = state.x
    key, new_state, reward, done, info = step(TETROMINOES, key, state, 0, env_config)
    assert new_state.x == initial_x - 1
    assert new_state.y == state.y + 1  # Due to gravity


def test_move_down(env_config, initial_state):
    key, state = initial_state
    initial_y = state.y
    key, new_state, reward, done, info = step(TETROMINOES, key, state, 2, env_config)
    assert new_state.y == initial_y + 2  # One for down action, one for gravity


def test_rotate_clockwise(env_config, initial_state):
    key, state = initial_state
    initial_rotation = state.rotation
    key, new_state, reward, done, info = step(TETROMINOES, key, state, 3, env_config)
    assert new_state.rotation == (initial_rotation + 1) % 4
    assert new_state.y == state.y + 1  # Due to gravity


def test_rotate_counterclockwise(env_config, initial_state):
    key, state = initial_state
    initial_rotation = state.rotation
    key, new_state, reward, done, info = step(TETROMINOES, key, state, 4, env_config)
    assert new_state.rotation == (initial_rotation - 1) % 4
    assert new_state.y == state.y + 1  # Due to gravity


def test_hard_drop(env_config, initial_state):
    key, state = initial_state
    key, new_state, reward, done, info = step(TETROMINOES, key, state, 6, env_config)
    assert new_state.y == 0  # Position for new tetromino shall be at the top
    assert (
        new_state.active_tetromino != state.active_tetromino
    )  # New tetromino should be spawned


def test_collision_detection(env_config, initial_state):
    key, state = initial_state
    # Place a block just below the initial position
    state = state.replace(board=state.board.at[state.y + 2, state.x].set(1))
    key, new_state, reward, done, info = step(
        TETROMINOES, key, state, 2, env_config
    )  # Try to move down
    assert new_state.y == 0  # New tetromino should be spawned
    assert (
        new_state.active_tetromino != state.active_tetromino
    )  # New tetromino should be spawned


def test_line_clearing(env_config, initial_state):
    key, state = initial_state
    # Fill a line
    state = state.replace(
        board=state.board.at[
            env_config.height - 1, env_config.padding : -env_config.padding
        ].set(1)
    )
    initial_score = state.score
    key, new_state, reward, done, info = step(
        TETROMINOES, key, state, 6, env_config
    )  # Hard drop to trigger line clear
    assert new_state.score > initial_score
    assert (
        jnp.sum(new_state.board[env_config.height - 1]) == 2 * env_config.padding
    )  # Only padding should remain


def test_game_over(env_config, initial_state):
    key, state = initial_state
    # Fill the board up to the top (leaving out one column so board doesn't clear)
    state = state.replace(
        board=state.board.at[
            : env_config.height, env_config.padding + 1 : -env_config.padding
        ].set(1)
    )
    key, new_state, reward, done, info = step(
        TETROMINOES, key, state, 6, env_config
    )  # Hard drop to trigger game over
    assert done


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
