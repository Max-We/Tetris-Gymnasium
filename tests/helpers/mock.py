import numpy as np


def generate_example_board_with_features(env):
    """
    Generate an example Tetris board with specific characteristics for testing.

    Args:
        env: The Tetris environment.

    Returns:
        A tuple containing:
        - board: The generated board.
        - height: Array of column heights.
        - max_height: Maximum height of the board.
        - holes: Number of holes in the board.
        - bumpiness: Measure of the board's bumpiness.
    """
    board = np.copy(env.unwrapped.board)
    # fill half the rows (except the last column to prevent line clear)
    target_height = env.unwrapped.height // 2
    board[
        target_height : env.unwrapped.height,
        env.unwrapped.padding : -(env.unwrapped.padding + 1),
    ] = 2
    # add some bumpiness
    board[target_height - 1, env.unwrapped.padding + 1] = 2
    board[target_height - 1, env.unwrapped.padding + 4] = 2
    board[target_height - 1, env.unwrapped.padding + 5] = 2
    # add some holes
    board[target_height + 2, env.unwrapped.padding + 2] = 0
    board[target_height + 4, env.unwrapped.padding + 3] = 0
    board[target_height + 6, env.unwrapped.padding + 6] = 0

    max_height = target_height + 1
    height = [target_height] * 10
    height[1], height[4], height[5], height[9] = max_height, max_height, max_height, 0
    holes = 3
    bumpiness = 14

    return (
        board,
        np.array(height, dtype=np.uint8),
        np.array([max_height], dtype=np.uint8),
        np.array([holes], dtype=np.uint8),
        np.array([bumpiness], dtype=np.uint8),
    )


def convert_to_base_observation(board):
    """Convert a board to a base observation like in the base-Tetris environment."""
    return {
        "board": board,
        "holder": None,
        "queue": None,
        "active_tetromino_mask": np.zeros_like(board),
    }
