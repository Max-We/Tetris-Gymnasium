from abc import abstractmethod

import numpy as np


class Randomizer:
    def __init__(self, size):
        self.size = size

    @abstractmethod
    def get_next_tetromino(self):
        pass


class BagRandomizer(Randomizer):
    def __init__(self, size):
        super().__init__(size)
        self.bag = np.arange(self.size, dtype=np.int8)
        self.index = 0
        self.shuffle_bag()

    def get_next_tetromino(self):
        tetromino_index = self.bag[self.index]
        self.index += 1
        # If we've reached the end of the bag, shuffle and start over
        if self.index >= len(self.bag):
            self.shuffle_bag()

        return tetromino_index

    def shuffle_bag(self):
        np.random.shuffle(self.bag)
        self.index = 0  # Reset index to the start


class RandomScheduler(Randomizer):
    def get_next_tetromino(self):
        return np.random.randint(0, self.size)
