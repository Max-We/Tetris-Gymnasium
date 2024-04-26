"""Randomizer classes for generating the order of tetrominoes in a game of Tetris."""
from abc import abstractmethod

import numpy as np
from gymnasium.utils.seeding import RandomNumberGenerator


class Randomizer:
    """Abstract class for tetromino randomizers."""

    def __init__(self, size: int):
        """Create a new randomizer with the given number of tetrominoes.

        A randomizer is an object that can be used to generate the order of tetrominoes in a game of Tetris.

        Args:
            size: The number of tetrominoes to choose from.
        """
        self.size = size
        self.rng: RandomNumberGenerator = None

    @abstractmethod
    def get_next_tetromino(self) -> int:
        """Get the index of the next tetromino to be used in the game.

        Returns: The index of the next tetromino to be used in the game.
        """
        pass

    @abstractmethod
    def reset(self, seed=None):
        """Resets the randomizer to start from a fresh state.

        This function is implemented after the usage pattern in Gymnasium, where seed is passed to the reset function
        only for the very first call after initialization. In all other cases, seed=None and the RNG is not reset.
        """
        if seed and seed > 0:
            # Passing a seed overwrites existing RNG
            seed_seq = np.random.SeedSequence(seed)
            self.rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
        elif self.rng is None:
            # If no seed is passed and no RNG has been created, create a default one
            self.rng = np.random.default_rng()


class BagRandomizer(Randomizer):
    """Randomly selects tetrominoes from a bag, ensuring that each tetromino is used once before reshuffling.

    The functionality is explained on the tetris wiki page: https://tetris.fandom.com/wiki/Random_Generator
    """

    def __init__(self, size):
        """Create a new bag randomizer with the given number of tetrominoes."""
        super().__init__(size)
        self.bag = np.arange(self.size, dtype=np.int8)
        self.index = 0

    def get_next_tetromino(self) -> int:
        """The bag randomizer returns the next tetromino in the bag.

        If the end of the bag is reached, the bag is reshuffled and the process starts over.

        Returns: The index of the next tetromino to be used in the game.
        """
        tetromino_index = self.bag[self.index]
        self.index += 1
        # If we've reached the end of the bag, shuffle and start over
        if self.index >= len(self.bag):
            self.shuffle_bag()

        return tetromino_index

    def shuffle_bag(self):
        """Shuffle the bag and reset the index to restart."""
        self.rng.shuffle(self.bag)
        self.index = 0  # Reset index to the start

    def reset(self, seed=None):
        """Resets the randomizer to start from a fresh state."""
        super().reset(seed)
        self.shuffle_bag()


class TrueRandomizer(Randomizer):
    """Randomly selects tetrominoes."""

    def __init__(self, size):
        """Create a new random scheduler with the given number of tetrominoes.

        Args:
            size: The number of tetrominoes to choose from.
        """
        super().__init__(size)

    def get_next_tetromino(self) -> int:
        """Return a random tetromino index."""
        return self.rng.randint(0, self.size)

    def reset(self, seed=None):
        """Resets the randomizer to start from a fresh state."""
        # In the case of `TrueRandomizer`, there is no state to reset
        super().reset(seed)
