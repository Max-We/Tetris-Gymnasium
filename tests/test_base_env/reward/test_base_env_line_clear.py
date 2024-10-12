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
        reward == (RewardsMapping.clear_line**2) * 10 + 1
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
