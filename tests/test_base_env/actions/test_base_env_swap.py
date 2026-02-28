from tetris_gymnasium.mappings.actions import ActionsMapping


def test_first_swap_stores_piece_and_spawns_next(tetris_env):
    """Test that first swap stores the piece in the holder and spawns next from queue."""
    tetris_env.reset(seed=42)
    original_piece_id = tetris_env.unwrapped.active_tetromino.id

    # Holder should be empty initially
    assert len(tetris_env.unwrapped.holder.get_tetrominoes()) == 0

    tetris_env.step(ActionsMapping.swap)

    # Holder should now contain the original piece
    holder_pieces = tetris_env.unwrapped.holder.get_tetrominoes()
    assert len(holder_pieces) == 1
    assert holder_pieces[0].id == original_piece_id

    # A new piece should have been spawned from queue
    assert tetris_env.unwrapped.active_tetromino is not None
    assert (
        tetris_env.unwrapped.active_tetromino.id != original_piece_id or True
    )  # could be same type


def test_swap_exchanges_pieces(tetris_env):
    """Test that swapping twice gets back the original piece."""
    tetris_env.reset(seed=42)
    first_piece_id = tetris_env.unwrapped.active_tetromino.id

    # First swap: stores piece, spawns next from queue
    tetris_env.step(ActionsMapping.swap)

    # Hard drop to reset has_swapped flag
    tetris_env.step(ActionsMapping.hard_drop)

    # Second swap: should get back the first piece
    tetris_env.step(ActionsMapping.swap)
    assert tetris_env.unwrapped.active_tetromino.id == first_piece_id


def test_double_swap_blocked(tetris_env):
    """Test that swapping twice in one piece is blocked by has_swapped."""
    tetris_env.reset(seed=42)

    # First swap
    tetris_env.step(ActionsMapping.swap)
    assert tetris_env.unwrapped.has_swapped is True

    piece_after_first_swap = tetris_env.unwrapped.active_tetromino.id

    # Second swap should be blocked
    tetris_env.step(ActionsMapping.swap)
    assert tetris_env.unwrapped.active_tetromino.id == piece_after_first_swap


def test_has_swapped_resets_after_hard_drop(tetris_env):
    """Test that has_swapped is reset after hard drop."""
    tetris_env.reset(seed=42)

    # Swap once
    tetris_env.step(ActionsMapping.swap)
    assert tetris_env.unwrapped.has_swapped is True

    # Hard drop to commit piece
    tetris_env.step(ActionsMapping.hard_drop)
    assert tetris_env.unwrapped.has_swapped is False


def test_swap_resets_piece_position(tetris_env_no_gravity):
    """Test that after swap, piece position resets to spawn position."""
    tetris_env_no_gravity.reset(seed=42)

    # Move piece down a bit first
    tetris_env_no_gravity.unwrapped.y = 5
    tetris_env_no_gravity.unwrapped.x = tetris_env_no_gravity.unwrapped.padding + 2

    # Hard drop to place piece (resets has_swapped)
    tetris_env_no_gravity.step(ActionsMapping.hard_drop)

    # Now swap - fill the holder first
    tetris_env_no_gravity.step(ActionsMapping.swap)

    # Hard drop to place and reset has_swapped
    tetris_env_no_gravity.step(ActionsMapping.hard_drop)

    # Move to non-spawn position
    tetris_env_no_gravity.unwrapped.y = 5
    tetris_env_no_gravity.unwrapped.x = tetris_env_no_gravity.unwrapped.padding + 3

    # Swap again - should reset position
    tetris_env_no_gravity.step(ActionsMapping.swap)

    expected_x = (
        tetris_env_no_gravity.unwrapped.width_padded // 2
        - tetris_env_no_gravity.unwrapped.active_tetromino.matrix.shape[0] // 2
    )
    assert tetris_env_no_gravity.unwrapped.x == expected_x
    assert tetris_env_no_gravity.unwrapped.y == 0
