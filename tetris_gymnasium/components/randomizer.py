from abc import abstractmethod

import numpy as np


class Randomizer:
    """Abstract class for tetromino randomizers."""
    def __init__(self, size):
        self.size = size

    @abstractmethod
    def get_next_tetromino(self):
        """Return the index of the next tetromino."""
        pass


class BagRandomizer(Randomizer):
    """Randomly selects tetrominoes from a bag."""
    def __init__(self, size):
        """Create a new bag randomizer with the given number of tetrominoes."""
        super().__init__(size)
        self.bag = np.arange(self.size, dtype=np.int8)
        self.index = 0
        self.shuffle_bag()

    def get_next_tetromino(self):
        """
        Tetris bag randomizer.
        Returns the next tetromino in the bag, and shuffles the bag once all elements have been drawn.
        """
        tetromino_index = self.bag[self.index]
        self.index += 1
        # If we've reached the end of the bag, shuffle and start over
        if self.index >= len(self.bag):
            self.shuffle_bag()

        return tetromino_index

    def shuffle_bag(self):
        """Shuffle the bag and reset the index to restart."""
        np.random.shuffle(self.bag)
        self.index = 0  # Reset index to the start


class RandomScheduler(Randomizer):
    """Randomly selects tetrominoes."""
    def __init__(self, size):
        super().__init__(size)
    def get_next_tetromino(self):
        """Return a random tetromino index."""
        return np.random.randint(0, self.size)
