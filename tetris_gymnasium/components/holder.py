"""Module for the Holder class, which stores one or more tetrominoes for later use in a game of Tetris."""
from collections import deque
from typing import Optional

import numpy as np


class Holder:
    """Class for one or more tetrominoes for later use in a game of Tetris."""

    def __init__(self, size=1):
        """Create a new holder with the given number of tetrominoes.

        Args:
            size: The number of tetrominoes to store. Defaults to 1.
        """
        self.size = size
        self.queue = deque(maxlen=size)

    def _get_tetromino(self) -> Optional[np.ndarray]:
        """Get the next tetromino from the holder."""
        return self.queue.popleft()

    def _store_tetromino(self, tetromino: np.ndarray):
        """Store a tetromino in the holder."""
        self.queue.append(tetromino)

    def swap(self, tetromino: np.ndarray) -> Optional[np.ndarray]:
        """Swap the given tetromino with the one in the holder.

        This implementation uses a queue to store the tetrominoes. Tetromioes are only returned once the queue is full.

        Args:
            tetromino: The tetromino to store in the holder.

        Returns:
            The tetromino that was in the holder before the swap.
        """
        if len(self.queue) < self.size:
            self._store_tetromino(tetromino)
            return None

        result = self._get_tetromino()
        self._store_tetromino(tetromino)
        return result
