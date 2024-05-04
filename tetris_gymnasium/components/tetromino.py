"""Data structures for Tetris."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Pixel:
    """A single pixel of a Tetris game.

    A pixel can be part of a tetromino or part of the game board (empty, bedrock).
    """

    id: int
    color_rgb: list


@dataclass
class Tetromino(Pixel):
    """A Tetris piece.

    A tetromino is a geometric shape composed of multiple pixels.
    """

    matrix: np.ndarray
