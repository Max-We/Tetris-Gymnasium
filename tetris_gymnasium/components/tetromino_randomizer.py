"""Randomizer classes for generating the order of tetrominoes in a game of Tetris."""
from abc import abstractmethod

import numpy as np
from gymnasium.utils.seeding import RandomNumberGenerator


class Randomizer:
    """Abstract class for tetromino randomizers.

    A randomizer is an object that can be used to generate the order of tetrominoes in a game of Tetris. When it's
    called via :func:`get_next_tetromino`, it returns the **index** of the next tetromino to be used in the game.
    This information can be used by the caller to get the actual tetromino object from a list of tetrominoes.
    """

    def __init__(self, size: int):
        """Create a randomizer for a specified number of tetrominoes to choose from.

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
        """Resets the randomizer.

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

    The bag randomizer is a common and popular approach in Tetris. It ensures that each tetromino is used once before
    reshuffling the bag, thus avoiding long sequences of the same tetromino.
    The functionality is explained on the tetris wiki page: https://tetris.fandom.com/wiki/Random_Generator
    """

    def __init__(self, size):
        """Create a new bag randomizer for a specified number of tetrominoes to choose from.

        Args:
            size: The number of tetrominoes to choose from.
        """
        super().__init__(size)
        self.bag = np.arange(self.size, dtype=np.int8)
        self.index = 0

    def get_next_tetromino(self) -> int:
        """Samples a new tetromino from the bag.

        Once the bag has been fully exploited, it is reshuffled and the process starts over.

        Returns: The index of the next tetromino to be used in the game.
        """
        tetromino_index = self.bag[self.index]
        self.index += 1
        # If we've reached the end of the bag, shuffle and start over
        if self.index >= len(self.bag):
            self.shuffle_bag()

        return tetromino_index

    def shuffle_bag(self):
        """Shuffle the bag and reset the index to restart the sampling process."""
        self.rng.shuffle(self.bag)
        self.index = 0  # Reset index to the start

    def reset(self, seed=None):
        """Resets the randomizer to start from a fresh state."""
        super().reset(seed)
        self.shuffle_bag()

    def __copy__(self):
        """Create a copy of the `BagRandomizer`."""
        new_randomizer = BagRandomizer(self.size)
        # RNG (this is faster than deepcopy)
        new_randomizer.rng = np.random.Generator(np.random.PCG64())
        new_randomizer.rng.bit_generator.state = self.rng.bit_generator.state
        # Content
        new_randomizer.bag = np.copy(self.bag.copy())
        new_randomizer.index = self.index
        return new_randomizer


class TrueRandomizer(Randomizer):
    """Randomly selects tetrominoes.

    This is the simplest form of randomizer, where each tetromino is chosen randomly. This approach can lead to
    sequences of the same tetromino, which may or may not be desired.
    """

    def __init__(self, size):
        """Create a new true randomizer for a specified number of tetrominoes to choose from.

        Args:
            size: The number of tetrominoes to choose from.
        """
        super().__init__(size)

    def get_next_tetromino(self) -> int:
        """Samples a new tetromino randomly."""
        return self.rng.randint(0, self.size)

    def reset(self, seed=None):
        """Resets the randomizer to start from a fresh state."""
        # In the case of `TrueRandomizer`, there is no state to reset
        # In this case, only the RNG is reset with the specified seed
        super().reset(seed)

    def __copy__(self):
        """Create a copy of the `TrueRandomizer`."""
        new_randomizer = TrueRandomizer(self.size)
        # RNG (this is faster than deepcopy)
        new_randomizer.rng = np.random.Generator(np.random.PCG64())
        new_randomizer.rng.bit_generator.state = self.rng.bit_generator.state
        return new_randomizer
