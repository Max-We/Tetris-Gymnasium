"""Tests for collision detection."""

from tetris_gymnasium.functional.core import collision, get_initial_x_y
from tetris_gymnasium.functional.tetrominoes import get_tetromino_matrix


class TestCollision:
    def test_no_collision_empty_board_center(
        self, empty_board, default_config, tetrominoes
    ):
        piece_idx = 0  # I-piece
        mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
        x, y = get_initial_x_y(default_config, tetrominoes, piece_idx)
        assert not collision(empty_board, mat, x, y)

    def test_collision_with_left_padding(self, empty_board, tetrominoes):
        mat = get_tetromino_matrix(tetrominoes, 0, 0)  # I-piece
        # Place at x=0, which overlaps left bedrock
        assert collision(empty_board, mat, 0, 0)

    def test_collision_with_bottom_padding(
        self, empty_board, default_config, tetrominoes
    ):
        mat = get_tetromino_matrix(tetrominoes, 0, 0)  # I-piece
        x, _ = get_initial_x_y(default_config, tetrominoes, 0)
        # Place at very bottom where bedrock is
        y = default_config.height  # into padding zone
        assert collision(empty_board, mat, x, y)

    def test_collision_with_placed_piece(
        self, empty_board, default_config, tetrominoes
    ):
        from tetris_gymnasium.functional.core import project_tetromino

        piece_idx = 2  # T-piece
        mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
        x, _ = get_initial_x_y(default_config, tetrominoes, piece_idx)
        y = 10
        tid = int(tetrominoes.ids[piece_idx])
        board = project_tetromino(empty_board, mat, x, y, tid)
        # Same position should now collide
        assert collision(board, mat, x, y)

    def test_no_collision_adjacent(self, empty_board, default_config, tetrominoes):
        from tetris_gymnasium.functional.core import project_tetromino

        piece_idx = 1  # O-piece (2x2 padded to 4x4)
        mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
        x, _ = get_initial_x_y(default_config, tetrominoes, piece_idx)
        y = 10
        tid = int(tetrominoes.ids[piece_idx])
        board = project_tetromino(empty_board, mat, x, y, tid)
        # Place another piece directly above
        y_above = y - 4  # far enough above the 4x4 matrix
        assert not collision(board, mat, x, y_above)
