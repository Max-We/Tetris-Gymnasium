from tetris_gymnasium.components.tetromino_randomizer import (
    BagRandomizer,
    TrueRandomizer,
)


def test_bag_randomizer_produces_all_7_pieces(bag_randomizer):
    """Test that first 7 pieces from BagRandomizer contain each ID exactly once."""
    pieces = [bag_randomizer.get_next_tetromino() for _ in range(7)]
    assert sorted(pieces) == list(range(7))


def test_bag_randomizer_same_seed_is_deterministic():
    """Test that same seed produces the same sequence."""
    r1 = BagRandomizer(7)
    r1.reset(seed=42)
    seq1 = [r1.get_next_tetromino() for _ in range(14)]

    r2 = BagRandomizer(7)
    r2.reset(seed=42)
    seq2 = [r2.get_next_tetromino() for _ in range(14)]

    assert seq1 == seq2


def test_bag_randomizer_second_bag_also_complete(bag_randomizer):
    """Test that the second bag of 7 also contains all pieces."""
    # Exhaust first bag
    for _ in range(7):
        bag_randomizer.get_next_tetromino()

    # Second bag
    pieces = [bag_randomizer.get_next_tetromino() for _ in range(7)]
    assert sorted(pieces) == list(range(7))


def test_true_randomizer_produces_valid_piece_ids(true_randomizer):
    """Test that TrueRandomizer only produces valid piece IDs in [0, 6]."""
    for _ in range(100):
        piece_id = true_randomizer.get_next_tetromino()
        assert 0 <= piece_id < 7


def test_true_randomizer_same_seed_is_deterministic():
    """Test that TrueRandomizer with same seed produces the same sequence."""
    r1 = TrueRandomizer(7)
    r1.reset(seed=42)
    seq1 = [r1.get_next_tetromino() for _ in range(20)]

    r2 = TrueRandomizer(7)
    r2.reset(seed=42)
    seq2 = [r2.get_next_tetromino() for _ in range(20)]

    assert seq1 == seq2
