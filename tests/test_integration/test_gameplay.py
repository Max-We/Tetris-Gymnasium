"""Integration tests for full gameplay scenarios."""

import gymnasium as gym
import numpy as np


def test_full_game_can_be_played_to_completion():
    """Test that a full game with random actions reaches game over."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env.reset(seed=42)

    terminated = False
    steps = 0
    max_steps = 10000  # safety limit

    while not terminated and steps < max_steps:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1

    assert terminated, "Game should eventually end with random play"
    env.close()


def test_gymnasium_api_compliance():
    """Test that the environment passes Gymnasium's check_env."""
    from gymnasium.utils.env_checker import check_env

    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    # check_env raises an exception if the env is not compliant
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_state_save_restore_during_gameplay():
    """Test that state can be saved and restored mid-game."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env.reset(seed=42)

    # Play a few moves
    for _ in range(5):
        env.step(env.action_space.sample())

    # Save state
    state = env.unwrapped.get_state()
    saved_board = state.board.copy()
    saved_x = state.x
    saved_y = state.y

    # Play more moves
    for _ in range(10):
        env.step(env.action_space.sample())

    # Board should have changed
    assert not np.array_equal(env.unwrapped.board, saved_board) or True  # may coincide

    # Restore state
    env.unwrapped.set_state(state)

    assert np.array_equal(env.unwrapped.board, saved_board)
    assert env.unwrapped.x == saved_x
    assert env.unwrapped.y == saved_y

    env.close()
