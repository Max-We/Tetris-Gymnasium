import numpy as np

from tetris_gymnasium.mappings.actions import ActionsMapping


def test_reset_produces_clean_board(tetris_env):
    """Test that reset gives an empty board after gameplay."""
    tetris_env.reset(seed=42)

    # Place some pieces
    tetris_env.step(ActionsMapping.hard_drop)
    tetris_env.step(ActionsMapping.hard_drop)

    # Reset
    tetris_env.reset(seed=42)
    board_after = tetris_env.unwrapped.board

    # Board should be clean (no placed pieces, only padding)
    playfield = board_after[
        : tetris_env.unwrapped.height,
        tetris_env.unwrapped.padding : -tetris_env.unwrapped.padding,
    ]
    assert np.all(playfield == 0), "Playfield should be empty after reset"


def test_reset_with_same_seed_is_deterministic(tetris_env):
    """Test that two resets with the same seed produce identical observations."""
    obs1, _ = tetris_env.reset(seed=42)
    obs2, _ = tetris_env.reset(seed=42)

    assert np.array_equal(obs1["board"], obs2["board"])
    assert np.array_equal(obs1["queue"], obs2["queue"])
    assert np.array_equal(obs1["holder"], obs2["holder"])


def test_reset_after_game_over_works(tetris_env):
    """Test that reset after game over returns a functional environment."""
    tetris_env.reset(seed=42)

    # Fill board to cause game over
    tetris_env.unwrapped.board[
        0 : tetris_env.unwrapped.height,
        tetris_env.unwrapped.padding : -tetris_env.unwrapped.padding,
    ] = 2
    tetris_env.step(ActionsMapping.hard_drop)
    assert tetris_env.unwrapped.game_over

    # Reset should work
    obs, info = tetris_env.reset(seed=42)
    assert not tetris_env.unwrapped.game_over
    assert obs is not None

    # Should be able to take actions
    obs, reward, terminated, truncated, info = tetris_env.step(ActionsMapping.no_op)
    assert obs is not None


def test_reset_clears_holder(tetris_env):
    """Test that reset clears the holder after a swap."""
    tetris_env.reset(seed=42)

    # Swap to fill the holder
    tetris_env.step(ActionsMapping.swap)
    assert len(tetris_env.unwrapped.holder.get_tetrominoes()) == 1

    # Reset
    tetris_env.reset(seed=42)

    # Holder should be empty
    assert len(tetris_env.unwrapped.holder.get_tetrominoes()) == 0
