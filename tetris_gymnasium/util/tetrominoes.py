"""Temporary file to store standard tetrominoes and their colors."""
import numpy as np

STANDARD_TETROMINOES = [
    np.array(
        [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    ),  # I
    np.array([[2, 2], [2, 2]], dtype=np.uint8),  # O
    np.array([[0, 3, 0], [3, 3, 3], [0, 0, 0]], dtype=np.uint8),  # T
    np.array([[0, 4, 4], [4, 4, 0], [0, 0, 0]], dtype=np.uint8),  # S
    np.array([[5, 5, 0], [0, 5, 5], [0, 0, 0]], dtype=np.uint8),  # Z
    np.array([[6, 0, 0], [6, 6, 6], [0, 0, 0]], dtype=np.uint8),  # L
    np.array([[0, 0, 7], [7, 7, 7], [0, 0, 0]], dtype=np.uint8),  # J
]

STANDARD_COLORS = np.array(
    [
        [0, 0, 0],  # Background color: black
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
