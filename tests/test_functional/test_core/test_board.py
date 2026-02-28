"""Tests for board creation and initial position."""

import jax.numpy as jnp


class TestCreateBoard:
    def test_board_shape(self, empty_board, default_config):
        # height + padding, width + 2*padding
        expected = (
            default_config.height + default_config.padding,
            default_config.width + 2 * default_config.padding,
        )
        assert empty_board.shape == expected  # (24, 18)

    def test_playable_area_is_zeros(self, empty_board, default_config):
        p = default_config.padding
        playable = empty_board[: default_config.height, p : p + default_config.width]
        assert jnp.all(playable == 0)

    def test_padding_is_bedrock(self, empty_board, default_config, tetrominoes):
        p = default_config.padding
        bedrock = tetrominoes.base_pixels[1]
        # Left padding
        assert jnp.all(empty_board[:, :p] == bedrock)
        # Right padding
        assert jnp.all(empty_board[:, p + default_config.width :] == bedrock)
        # Bottom padding
        assert jnp.all(empty_board[default_config.height :, :] == bedrock)


class TestGetInitialXY:
    def test_centers_piece(self, default_config, tetrominoes):
        from tetris_gymnasium.functional.core import get_initial_x_y

        for piece_idx in range(7):
            x, y = get_initial_x_y(default_config, tetrominoes, piece_idx)
            assert y == 0
            # x should center the piece in the padded board
            board_width = default_config.width + 2 * default_config.padding
            from tetris_gymnasium.functional.tetrominoes import get_tetromino_matrix

            mat = get_tetromino_matrix(tetrominoes, piece_idx, 0)
            expected_x = board_width // 2 - mat.shape[1] // 2
            assert int(x) == int(expected_x)
