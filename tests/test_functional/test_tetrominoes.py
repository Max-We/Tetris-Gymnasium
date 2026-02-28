"""Tests for tetromino definitions and lookup."""

import jax.numpy as jnp

from tetris_gymnasium.functional.tetrominoes import (
    TETROMINOES,
    get_tetromino_color,
    get_tetromino_matrix,
)


class TestTetrominoes:
    def test_seven_tetrominoes_with_unique_ids(self):
        assert TETROMINOES.ids.shape == (7,)
        ids = set(TETROMINOES.ids.tolist())
        assert ids == {2, 3, 4, 5, 6, 7, 8}

    def test_matrices_shape(self):
        assert TETROMINOES.matrices.shape == (7, 4, 4, 4)

    def test_get_tetromino_matrix_returns_4x4(self):
        for piece in range(7):
            for rot in range(4):
                mat = get_tetromino_matrix(TETROMINOES, piece, rot)
                assert mat.shape == (4, 4)

    def test_o_piece_rotation_invariant(self):
        # O-piece is index 1
        m0 = get_tetromino_matrix(TETROMINOES, 1, 0)
        for rot in range(1, 4):
            m = get_tetromino_matrix(TETROMINOES, 1, rot)
            assert jnp.array_equal(m0, m)

    def test_four_rotations_cycle(self):
        for piece in range(7):
            m0 = get_tetromino_matrix(TETROMINOES, piece, 0)
            m4 = get_tetromino_matrix(TETROMINOES, piece, 0)
            assert jnp.array_equal(m0, m4)

    def test_get_tetromino_color(self):
        expected_colors = [
            (0, 240, 240),  # I  (index 0, id 2)
            (240, 240, 0),  # O  (index 1, id 3)
            (160, 0, 240),  # T  (index 2, id 4)
            (0, 240, 0),  # S  (index 3, id 5)
            (240, 0, 0),  # Z  (index 4, id 6)
            (0, 0, 240),  # J  (index 5, id 7)
            (240, 160, 0),  # L  (index 6, id 8)
        ]
        for i, expected in enumerate(expected_colors):
            color = get_tetromino_color(TETROMINOES, i)
            # int8 wraps, compare as uint8
            assert tuple(color.astype(jnp.uint8).tolist()) == expected

    def test_base_pixels(self):
        assert jnp.array_equal(
            TETROMINOES.base_pixels, jnp.array([0, 1], dtype=jnp.int8)
        )
        assert TETROMINOES.base_pixel_colors.shape == (2, 3)
