from tetris_gymnasium.components.tetromino_queue import TetrominoQueue
from tetris_gymnasium.components.tetromino_randomizer import BagRandomizer


def test_queue_maintains_size(tetromino_queue):
    """Test that queue maintains its size after getting next piece."""
    initial_size = tetromino_queue.size
    assert len(tetromino_queue.get_queue()) == initial_size

    tetromino_queue.get_next_tetromino()

    assert len(tetromino_queue.get_queue()) == initial_size


def test_queue_returns_fifo_order(tetromino_queue):
    """Test that first piece out matches first piece generated."""
    # After reset, the queue has 4 pieces; first piece should be what was generated first
    queue_contents = tetromino_queue.get_queue()
    first_in_queue = queue_contents[0]

    first_out = tetromino_queue.get_next_tetromino()

    assert first_out == first_in_queue


def test_queue_reset_with_seed_is_deterministic():
    """Test that same seed produces the same sequence."""
    randomizer1 = BagRandomizer(7)
    queue1 = TetrominoQueue(randomizer1, size=4)
    queue1.reset(seed=42)
    seq1 = [queue1.get_next_tetromino() for _ in range(10)]

    randomizer2 = BagRandomizer(7)
    queue2 = TetrominoQueue(randomizer2, size=4)
    queue2.reset(seed=42)
    seq2 = [queue2.get_next_tetromino() for _ in range(10)]

    assert seq1 == seq2
