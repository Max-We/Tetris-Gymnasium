"""Tests for check_game_over."""

from tetris_gymnasium.functional.core import (
    check_game_over,
    create_board,
    get_initial_x_y,
    project_tetromino,
)
from tetris_gymnasium.functional.tetrominoes import get_tetromino_matrix


class TestCheckGameOver:
    def test_no_game_over_empty_board(self, default_config, tetrominoes):
        board = create_board(default_config, tetrominoes)
        piece_idx = 0
        x, y = get_initial_x_y(default_config, tetrominoes, piece_idx)
        assert not check_game_over(tetrominoes, board, piece_idx, 0, x, y)

    def test_game_over_when_spawn_blocked(self, default_config, tetrominoes):
        board = create_board(default_config, tetrominoes)
        piece_idx = 0
        x, y = get_initial_x_y(default_config, tetrominoes, piece_idx)
        mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
        tid = int(tetrominoes.ids[piece_idx])
        # Fill the spawn position
        board = project_tetromino(board, mat, x, y, tid)
        assert check_game_over(tetrominoes, board, piece_idx, 0, x, y)
