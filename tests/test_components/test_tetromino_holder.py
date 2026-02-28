import copy

import numpy as np

from tetris_gymnasium.components.tetromino import Tetromino


def _make_tetromino(tid):
    """Helper to create a simple tetromino with given id."""
    return Tetromino(tid, [0, 0, 0], np.array([[1, 1], [1, 1]], dtype=np.uint8))


def test_empty_holder_returns_none_on_first_swap(tetromino_holder):
    """Test that swap returns None when holder is empty."""
    piece = _make_tetromino(0)
    result = tetromino_holder.swap(piece)
    assert result is None


def test_full_holder_returns_stored_piece(tetromino_holder):
    """Test that after first swap fills it, second swap returns the original piece."""
    piece_a = _make_tetromino(0)
    piece_b = _make_tetromino(1)

    tetromino_holder.swap(piece_a)  # fills holder, returns None
    result = tetromino_holder.swap(piece_b)  # returns piece_a

    assert result is not None
    assert result.id == piece_a.id


def test_reset_clears_holder(tetromino_holder):
    """Test that reset clears the holder after storing a piece."""
    piece = _make_tetromino(0)
    tetromino_holder.swap(piece)
    assert len(tetromino_holder.get_tetrominoes()) == 1

    tetromino_holder.reset()

    assert len(tetromino_holder.get_tetrominoes()) == 0


def test_copy_creates_independent_instance(tetromino_holder):
    """Test that modifying a copy doesn't affect the original."""
    piece = _make_tetromino(0)
    tetromino_holder.swap(piece)

    holder_copy = copy.copy(tetromino_holder)

    # Modify the copy
    holder_copy.reset()

    # Original should still have the piece
    assert len(tetromino_holder.get_tetrominoes()) == 1
    assert len(holder_copy.get_tetrominoes()) == 0
