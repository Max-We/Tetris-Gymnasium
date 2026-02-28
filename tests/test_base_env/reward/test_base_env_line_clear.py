import copy

import numpy as np

from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_gymnasium.mappings.rewards import RewardsMapping


def test_line_clear_normal(tetris_env, vertical_i_tetromino):
    """Test normal line clear with vertical I-tetromino."""
    tetris_env.reset(seed=42)
    cleared_board = np.copy(tetris_env.unwrapped.board)

    # Fill board except the last column
    tetris_env.unwrapped.board[
        tetris_env.unwrapped.height
        - vertical_i_tetromino.matrix.shape[0] : -tetris_env.unwrapped.padding,
        tetris_env.unwrapped.padding : -tetris_env.unwrapped.padding - 1,
    ] = 2
    assert np.any(tetris_env.unwrapped.board != cleared_board)

    # Set I-tetromino and move to the last column
    tetris_env.unwrapped.active_tetromino = vertical_i_tetromino
    tetris_env.unwrapped.x = (
        tetris_env.unwrapped.width + tetris_env.unwrapped.padding - 2
    )

    # Lock in tetromino
    observation, reward, terminated, truncated, info = tetris_env.step(
        ActionsMapping.hard_drop
    )

    # Check that the board has been cleared and that the reward is correct
    assert np.array_equal(tetris_env.unwrapped.board, cleared_board)
    assert (
        reward == ((RewardsMapping.clear_line * 4) ** 2) * 10 + 1
    )  # In the future, this reward formula may change

    # Check that the game is not terminated or truncated
    assert not terminated and not truncated

    # Check that a new tetromino has been spawned and x, y have been reset
    assert tetris_env.unwrapped.active_tetromino is not None
    assert (
        tetris_env.unwrapped.x
        == tetris_env.unwrapped.width_padded // 2
        - tetris_env.unwrapped.active_tetromino.matrix.shape[0] // 2
    )
    assert tetris_env.unwrapped.y == 0


def test_multi_line_clear_two_lines(tetris_env):
    """Test clearing 2 lines simultaneously."""
    tetris_env.reset(seed=42)

    padding = tetris_env.unwrapped.padding

    # Fill the bottom 2 rows completely except the last 2 columns
    tetris_env.unwrapped.board[
        tetris_env.unwrapped.height - 2 : tetris_env.unwrapped.height,
        padding : -padding - 2,
    ] = 2

    # Use O-tetromino (2x2) from the env's tetrominoes to complete the last 2 columns
    o_tetromino = copy.deepcopy(tetris_env.unwrapped.tetrominoes[1])
    tetris_env.unwrapped.active_tetromino = o_tetromino
    tetris_env.unwrapped.x = tetris_env.unwrapped.width + padding - 2

    observation, reward, terminated, truncated, info = tetris_env.step(
        ActionsMapping.hard_drop
    )

    # Should have cleared 2 lines
    assert info["lines_cleared"] == 2
    assert reward == (2**2) * tetris_env.unwrapped.width + RewardsMapping.alife


def test_no_line_clear_when_row_incomplete(tetris_env):
    """Test that no lines are cleared when no row is complete."""
    tetris_env.reset(seed=42)

    # Place piece without completing any row
    observation, reward, terminated, truncated, info = tetris_env.step(
        ActionsMapping.hard_drop
    )

    assert info["lines_cleared"] == 0
    # Reward should be alife only (no line clear bonus)
    assert reward == RewardsMapping.alife
