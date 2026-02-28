"""Tests for hard_drop core function."""

from tetris_gymnasium.functional.core import (
    get_initial_x_y,
    hard_drop,
    project_tetromino,
)
from tetris_gymnasium.functional.tetrominoes import get_tetromino_matrix


class TestHardDrop:
    def test_drops_to_bottom_empty_board(
        self, empty_board, default_config, tetrominoes
    ):
        piece_idx = 0  # I-piece
        mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
        x, y = get_initial_x_y(default_config, tetrominoes, piece_idx)
        new_y, reward = hard_drop(empty_board, mat, x, y)
        assert int(new_y) > int(y)
        # Should be near the bottom
        assert int(new_y) >= default_config.height - 4

    def test_stops_on_existing_piece(self, empty_board, default_config, tetrominoes):
        piece_idx = 2  # T-piece
        mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
        x, _ = get_initial_x_y(default_config, tetrominoes, piece_idx)
        tid = int(tetrominoes.ids[piece_idx])
        # Place a piece at y=15
        board = project_tetromino(empty_board, mat, x, 15, tid)
        # Drop another T from top
        new_y, _ = hard_drop(board, mat, x, 0)
        assert int(new_y) < 15

    def test_reward_is_twice_distance(self, empty_board, default_config, tetrominoes):
        piece_idx = 0
        mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
        x, y = get_initial_x_y(default_config, tetrominoes, piece_idx)
        new_y, reward = hard_drop(empty_board, mat, x, y)
        assert int(reward) == 2 * (int(new_y) - int(y))
