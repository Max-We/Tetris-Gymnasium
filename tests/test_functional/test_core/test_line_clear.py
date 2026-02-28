"""Tests for clear_filled_rows."""

import jax.numpy as jnp

from tetris_gymnasium.functional.core import clear_filled_rows, create_board


class TestClearFilledRows:
    def test_no_clear_empty_board(self, default_config, tetrominoes, empty_board):
        board, n_filled = clear_filled_rows(default_config, tetrominoes, empty_board)
        assert int(n_filled) == 0
        assert jnp.array_equal(board, empty_board)

    def test_single_row_clear(self, default_config, tetrominoes):
        board = create_board(default_config, tetrominoes)
        p = default_config.padding
        # Fill the bottom playable row
        row = default_config.height - 1
        board = board.at[row, p : p + default_config.width].set(2)
        new_board, n_filled = clear_filled_rows(default_config, tetrominoes, board)
        assert int(n_filled) == 1
        # Bottom row should be cleared (all zeros)
        playable = new_board[
            default_config.height - 1,
            p : p + default_config.width,
        ]
        assert jnp.all(playable == 0)

    def test_double_row_clear(self, default_config, tetrominoes):
        board = create_board(default_config, tetrominoes)
        p = default_config.padding
        for row in [
            default_config.height - 1,
            default_config.height - 2,
        ]:
            board = board.at[row, p : p + default_config.width].set(2)
        _, n_filled = clear_filled_rows(default_config, tetrominoes, board)
        assert int(n_filled) == 2

    def test_quad_row_clear(self, default_config, tetrominoes):
        board = create_board(default_config, tetrominoes)
        p = default_config.padding
        for i in range(4):
            row = default_config.height - 1 - i
            board = board.at[row, p : p + default_config.width].set(2)
        _, n_filled = clear_filled_rows(default_config, tetrominoes, board)
        assert int(n_filled) == 4

    def test_partial_row_not_cleared(self, default_config, tetrominoes):
        board = create_board(default_config, tetrominoes)
        p = default_config.padding
        row = default_config.height - 1
        # Fill all but one column
        board = board.at[row, p : p + default_config.width - 1].set(2)
        _, n_filled = clear_filled_rows(default_config, tetrominoes, board)
        assert int(n_filled) == 0

    def test_rows_shift_down_after_clear(self, default_config, tetrominoes):
        board = create_board(default_config, tetrominoes)
        p = default_config.padding
        # Place a marker in a row above
        marker_row = default_config.height - 3
        board = board.at[marker_row, p].set(5)
        # Fill the bottom row
        board = board.at[
            default_config.height - 1,
            p : p + default_config.width,
        ].set(2)
        new_board, _ = clear_filled_rows(default_config, tetrominoes, board)
        # Marker should have shifted down by 1
        assert int(new_board[marker_row + 1, p]) == 5
