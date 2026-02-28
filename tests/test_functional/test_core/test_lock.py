"""Tests for lock_active_tetromino."""

import jax.numpy as jnp

from tetris_gymnasium.functional.core import (
    create_board,
    get_initial_x_y,
    lock_active_tetromino,
    score,
)


class TestLockActiveTetromino:
    def test_board_contains_tetromino_after_lock(self, default_config, tetrominoes):
        board = create_board(default_config, tetrominoes)
        piece_idx = 2  # T-piece
        x, _ = get_initial_x_y(default_config, tetrominoes, piece_idx)
        y = 10
        rotation = 0
        new_board, reward, lines = lock_active_tetromino(
            default_config, tetrominoes, board, piece_idx, rotation, x, y
        )
        tid = int(tetrominoes.ids[piece_idx])
        # Board should contain the tetromino id somewhere
        assert jnp.any(new_board == tid)

    def test_filled_rows_cleared(self, default_config, tetrominoes):
        board = create_board(default_config, tetrominoes)
        p = default_config.padding
        # Fill a row completely to guarantee a clear
        board = board.at[
            default_config.height - 1,
            p : p + default_config.width,
        ].set(2)
        # Lock an I-piece that doesn't overlap the filled row
        piece_idx = 2  # T-piece
        x, _ = get_initial_x_y(default_config, tetrominoes, piece_idx)
        _, reward, lines = lock_active_tetromino(
            default_config,
            tetrominoes,
            board,
            piece_idx,
            0,
            x,
            5,
        )
        # Pre-filled row should be cleared
        assert int(lines) >= 1

    def test_reward_matches_score(self, default_config, tetrominoes):
        board = create_board(default_config, tetrominoes)
        p = default_config.padding
        # Fill bottom row for a guaranteed line clear
        board = board.at[
            default_config.height - 1,
            p : p + default_config.width,
        ].set(2)
        piece_idx = 2
        x, _ = get_initial_x_y(default_config, tetrominoes, piece_idx)
        _, reward, lines = lock_active_tetromino(
            default_config, tetrominoes, board, piece_idx, 0, x, 5
        )
        expected = score(default_config, lines)
        assert int(reward) == int(expected)
