"""Tests for project_tetromino."""

import jax.numpy as jnp

from tetris_gymnasium.functional.core import get_initial_x_y, project_tetromino
from tetris_gymnasium.functional.tetrominoes import get_tetromino_matrix


class TestProjectTetromino:
    def test_places_tetromino_id_at_position(
        self, empty_board, default_config, tetrominoes
    ):
        piece_idx = 2  # T-piece
        mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
        x, y = get_initial_x_y(default_config, tetrominoes, piece_idx)
        tid = int(tetrominoes.ids[piece_idx])
        board = project_tetromino(empty_board, mat, x, y, tid)
        # Where mat > 0, board should have tid
        h, w = mat.shape
        section = board[y : y + h, x : x + w]
        expected_vals = jnp.where(mat > 0, tid, 0)
        assert jnp.array_equal(section, expected_vals)

    def test_preserves_existing_values(self, empty_board, default_config, tetrominoes):
        piece_idx = 0  # I-piece
        mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
        x, _ = get_initial_x_y(default_config, tetrominoes, piece_idx)
        tid = int(tetrominoes.ids[piece_idx])
        board1 = project_tetromino(empty_board, mat, x, 5, tid)

        piece_idx2 = 2  # T-piece
        mat2 = get_tetromino_matrix(tetrominoes, piece_idx2, 0)
        tid2 = int(tetrominoes.ids[piece_idx2])
        board2 = project_tetromino(board1, mat2, x, 10, tid2)

        # Original I-piece values should still be there
        h, w = mat.shape
        section = board2[5 : 5 + h, x : x + w]
        expected = jnp.where(mat > 0, tid, 0)
        assert jnp.array_equal(section, expected)

    def test_zero_cells_dont_affect_board(
        self, empty_board, default_config, tetrominoes
    ):
        piece_idx = 2  # T-piece has zeros in matrix
        mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
        x, y = get_initial_x_y(default_config, tetrominoes, piece_idx)
        tid = int(tetrominoes.ids[piece_idx])
        board = project_tetromino(empty_board, mat, x, y, tid)
        h, w = mat.shape
        section = board[y : y + h, x : x + w]
        # Where mat == 0, board should remain 0
        zero_mask = mat == 0
        assert jnp.all(section[zero_mask] == 0)
