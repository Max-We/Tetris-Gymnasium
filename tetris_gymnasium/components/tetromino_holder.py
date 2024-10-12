"""Module for the Holder class, which stores one or more tetrominoes for later use in a game of Tetris."""
from collections import deque
from typing import Optional

from tetris_gymnasium.components.tetromino import Tetromino


class TetrominoHolder:
    """A holder can store one or more tetrominoes for later use in a game of Tetris.

    Tetrominoes can be swapped in- and out during the game.
    """

    def __init__(self, size=1):
        """Create a new holder with the given number of tetrominoes.

        Args:
            size: The number of tetrominoes to store. Defaults to 1.
        """
        self.size = size
        self.queue = deque(maxlen=size)

    def _get_tetromino(self) -> Optional[Tetromino]:
        """Get the next tetromino from the holder."""
        return self.queue.popleft()

    def _store_tetromino(self, tetromino: Tetromino):
        """Store a tetromino in the holder."""
        self.queue.append(tetromino)

    def swap(self, tetromino: Tetromino) -> Optional[Tetromino]:
        """Swap the given tetromino with the one in the holder.

        This implementation uses a queue to store the tetrominoes. Tetrominoes are only returned once the queue is full.
        If this is not the case, the provided tetromino is stored in the queue and None is returned.

        Args:
            tetromino: The tetromino to store in the holder.

        Returns:
            The oldest tetromino that's stored in the queue, if the queue is full. Otherwise, None.
        """
        if len(self.queue) < self.size:
            self._store_tetromino(tetromino)
            return None

        result = self._get_tetromino()
        self._store_tetromino(tetromino)
        return result

    def reset(self):
        """Reset the holder to its initial state. This involves clearing the queue."""
        self.queue.clear()

    def get_tetrominoes(self):
        """Get all the tetrominoes currently in the holder."""
        return list(self.queue)

    def __copy__(self):
        """Create a copy of the holder."""
        new_holder = TetrominoHolder(self.size)
        new_holder.queue = deque(self.queue)
        return new_holder
