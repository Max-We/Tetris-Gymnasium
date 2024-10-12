"""Data structures for Tetris."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Pixel:
    """A single pixel in a game of Tetris.

    A pixel is the basic building block of the game and has an id and a color.

    The basic pixels are by default the empty pixel (id=0) and the bedrock pixel (id=1).
    Additionally, multiple pixels can be combined to form a tetromino.
    """

    id: int
    color_rgb: list


@dataclass
class Tetromino(Pixel):
    """A Tetris "piece" is called a Tetromino. Examples are the I, J, L, O, S, T, and Z pieces.

    On a conceptual basis, a tetromino is a 2D-array composed of multiple pixels. All pixels that compose the tetromino
    have the same id. And the ids of all the pixels are stored in the matrix.

    An example for the matrix of the T-tetromino:

    .. code-block:: python

        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0]
        ]

    In the matrix, the value `0` represents an empty pixel, and the value `1` represents a pixel of the T-tetromino.

    When initializing a `Tetromino` object on your own, you'll typically use binary values for the matrix, where `1`
    represents a pixel of the tetromino and `0` represents an empty pixel.
    """

    matrix: np.ndarray

    def __copy__(self):
        """Create a copy of the tetromino."""
        return Tetromino(
            id=self.id,
            color_rgb=self.color_rgb.copy(),
            matrix=self.matrix.copy(),
        )
