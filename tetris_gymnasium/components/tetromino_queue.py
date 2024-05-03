"""Module for a queue of tetrominoes for use in a game of Tetris."""
from collections import deque

from tetris_gymnasium.components.tetromino_randomizer import Randomizer


class TetrominoQueue:
    """Class for a queue of tetrominoes for use in a game of Tetris."""

    def __init__(self, randomizer: Randomizer, size=4):
        """Create a new queue of tetrominoes with the given size.

        Args:
            randomizer: The randomizer that generates the tetrominoes sequence.
            size: The number of tetrominoes to store. Defaults to 4.
        """
        self.randomizer = randomizer
        self.queue = deque(maxlen=size)
        self.size = size

    def reset(self, seed=None):
        """Reset the queue to its initial state."""
        self.randomizer.reset(seed)
        self.queue.clear()
        for _ in range(self.size):
            self.queue.append(self.randomizer.get_next_tetromino())

    def get_next_tetromino(self):
        """Get the next tetromino from the queue and generates a new one."""
        tetromino = self.queue.popleft()
        self.queue.append(self.randomizer.get_next_tetromino())
        return tetromino

    def get_queue(self):
        """Get the tetrominoes currently in the queue."""
        return list(self.queue)
