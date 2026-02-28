"""Tests for gravity step."""

from tetris_gymnasium.functional.core import (
    get_initial_x_y,
    graviy_step,
    project_tetromino,
)
from tetris_gymnasium.functional.tetrominoes import get_tetromino_matrix


class TestGravityStep:
    def test_moves_down_on_empty_board(self, empty_board, default_config, tetrominoes):
        piece_idx = 0
        x, y = get_initial_x_y(default_config, tetrominoes, piece_idx)
        new_y = graviy_step(tetrominoes, empty_board, piece_idx, 0, x, y)
        assert int(new_y) == int(y) + 1

    def test_blocked_at_bottom(self, empty_board, default_config, tetrominoes):
        piece_idx = 0
        get_tetromino_matrix(tetrominoes, piece_idx, 0)
        x, _ = get_initial_x_y(default_config, tetrominoes, piece_idx)
        # I-piece row 1 has blocks; bottom of playable is height-1
        # Place as far down as possible without collision
        y = default_config.height - 2  # near bottom bedrock
        new_y = graviy_step(tetrominoes, empty_board, piece_idx, 0, x, y)
        assert int(new_y) == int(y)  # blocked

    def test_blocked_by_placed_piece(self, empty_board, default_config, tetrominoes):
        piece_idx = 2  # T-piece
        mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
        x, _ = get_initial_x_y(default_config, tetrominoes, piece_idx)
        tid = int(tetrominoes.ids[piece_idx])
        # Place a piece at y=10
        board = project_tetromino(empty_board, mat, x, 10, tid)
        # Piece above at y=7 should be able to move down
        y = 7
        new_y = graviy_step(tetrominoes, board, piece_idx, 0, x, y)
        assert int(new_y) == 8
        # But at y=8, next step should be blocked
        # Try at y just above the placed piece
        new_y2 = graviy_step(tetrominoes, board, piece_idx, 0, x, 8)
        # T at (x,10) occupies rows 10-12; T at (x,9) overlaps
        assert int(new_y2) == 8  # blocked
