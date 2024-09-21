"""Module for a queue of tetrominoes for use in a game of Tetris."""
from collections import deque
from copy import deepcopy

from tetris_gymnasium.components.tetromino_randomizer import Randomizer


class TetrominoQueue:
    """The `TetrominoQueue` stores all incoming tetrominoes in a queue.

    The sequence of pieces is generated by a :class:`Randomizer`, which can be customized by the user.
    """

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
        """Reset the queue to its initial state.

        Args:
            seed: The seed to use for the randomizer. Defaults to None.
        """
        self.randomizer.reset(seed)
        self.queue.clear()
        for _ in range(self.size):
            self.queue.append(self.randomizer.get_next_tetromino())

    def get_next_tetromino(self):
        """Gets the next tetromino from the queue and generates a new one.

        Generating a new Tetromino makes sure that the queue will always be full.
        """
        tetromino = self.queue.popleft()
        self.queue.append(self.randomizer.get_next_tetromino())
        return tetromino

    def get_queue(self):
        """Get all tetrominoes currently in the queue."""
        return list(self.queue)

    def __deepcopy__(self, memo):
        # Create a new instance
        new_queue = TetrominoQueue(deepcopy(self.randomizer, memo), self.size)

        # Deep copy the queue
        new_queue.queue = deepcopy(self.queue, memo)

        return new_queue
