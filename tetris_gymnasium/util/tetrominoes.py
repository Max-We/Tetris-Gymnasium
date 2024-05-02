"""Temporary file to store standard tetrominoes and their colors."""
from typing import List

import numpy as np


def pre_process(tetrominos: List[np.ndarray]):
    """In order to make the tetrominos distinguishable, each tetromino should have a unique value.

    The values start at 2, as 0 is the background and 1 is the bedrock.

    Args:
        tetrominos: The masks of the tetrominos.

    Returns:
        The preprocessed tetrominos.
    """
    for i in range(len(tetrominos)):
        tetrominos[i] = tetrominos[i] * (i + 2)
    return tetrominos


STANDARD_TETROMINOES_MASKS = [
    np.array(
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    ),  # I
    np.array([[1, 1], [1, 1]], dtype=np.uint8),  # O
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),  # T
    np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.uint8),  # S
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8),  # Z
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),  # L
    np.array([[0, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),  # J
]

STANDARD_TETROMINOES = pre_process(STANDARD_TETROMINOES_MASKS)

STANDARD_COLORS = np.array(
    [
        [0, 0, 0],  # Background color: Black
        [128, 128, 128],  # Bedrock: Grey
        [0, 240, 240],  # I: Cyan
        [240, 240, 0],  # O: Yellow
        [160, 0, 240],  # T: Purple
        [0, 240, 0],  # S: Green
        [240, 0, 0],  # Z: Red
        [0, 0, 240],  # J: Blue
        [240, 160, 0],  # L: Orange
    ],
    dtype=np.uint8,
)
