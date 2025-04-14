def test_grouped_termination_out_of_border(tetris_env_grouped):
    """Test that the environment terminates when a tetromino is placed outside the border"""
    observation, reward, terminated, truncated, info = tetris_env_grouped.step(
        39
    )  # illegal action: tetromino outside matrix
    assert terminated


def test_grouped_termination_blocked_by_tetrominoes(tetris_env_grouped):
    """Test that the environment terminates when a tetromino is placed on top of another tetromino and the board is filled with tetrominoes"""
    # Fill the board with tetrominoes (except last col to prevent line clear)
    tetris_env_grouped.unwrapped.board[
        1 : -tetris_env_grouped.unwrapped.padding,
        tetris_env_grouped.unwrapped.padding : -(
            tetris_env_grouped.unwrapped.padding - 1
        ),
    ] = 2

    # Try placing another one on top
    observation, reward, terminated, truncated, info = tetris_env_grouped.step(
        0
    )  # illegal action: tetromino outside matrix
    assert terminated
