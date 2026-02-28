"""Fixtures for component tests."""

import pytest

from tetris_gymnasium.components.tetromino_holder import TetrominoHolder
from tetris_gymnasium.components.tetromino_queue import TetrominoQueue
from tetris_gymnasium.components.tetromino_randomizer import (
    BagRandomizer,
    TrueRandomizer,
)


@pytest.fixture
def bag_randomizer():
    """Fixture for a BagRandomizer with 7 pieces."""
    randomizer = BagRandomizer(7)
    randomizer.reset(seed=42)
    return randomizer


@pytest.fixture
def true_randomizer():
    """Fixture for a TrueRandomizer with 7 pieces."""
    randomizer = TrueRandomizer(7)
    randomizer.reset(seed=42)
    return randomizer


@pytest.fixture
def tetromino_queue(bag_randomizer):
    """Fixture for a TetrominoQueue with default size."""
    queue = TetrominoQueue(bag_randomizer, size=4)
    queue.reset(seed=42)
    return queue


@pytest.fixture
def tetromino_holder():
    """Fixture for a TetrominoHolder with default size."""
    return TetrominoHolder(size=1)
