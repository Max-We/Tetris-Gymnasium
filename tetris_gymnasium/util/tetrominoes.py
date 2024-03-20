"""Temporary file to store standard tetrominoes and their colors."""
import numpy as np

STANDARD_TETROMINOES = [
    np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
    np.array([[2, 2], [2, 2]], dtype=np.uint8),
    np.array([[0, 3, 0], [3, 3, 3], [0, 0, 0]], dtype=np.uint8),
    np.array([[0, 4, 4], [4, 4, 0], [0, 0, 0]], dtype=np.uint8),
    np.array([[5, 5, 0], [0, 5, 5], [0, 0, 0]], dtype=np.uint8),
    np.array([[6, 0, 0], [6, 6, 6], [0, 0, 0]], dtype=np.uint8),
    np.array([[0, 0, 7], [7, 7, 7], [0, 0, 0]], dtype=np.uint8),
]

STANDARD_COLORS = [
    (0, 0, 0),
    (255, 255, 0),
    (147, 88, 254),
    (54, 175, 144),
    (255, 0, 0),
    (102, 217, 238),
    (254, 151, 32),
    (0, 0, 255),
]
