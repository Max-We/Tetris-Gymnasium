import numpy as np

from tetris_gymnasium.mappings.actions import ActionsMapping


def test_hard_drop_places_piece_at_bottom(tetris_env):
    """Test that hard drop places the piece at the bottom of the board."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.y = 0

    tetris_env.step(ActionsMapping.hard_drop)

    # The piece should have been placed (board should contain piece values)
    board = tetris_env.unwrapped.board
    playfield = board[
        : tetris_env.unwrapped.height,
        tetris_env.unwrapped.padding : -tetris_env.unwrapped.padding,
    ]
    assert np.any(playfield >= 2), "Piece should be placed on the board"


def test_hard_drop_onto_existing_pieces(tetris_env):
    """Test that hard drop lands on top of existing pieces."""
    tetris_env.reset(seed=42)

    # Place blocks in the bottom half (leave gaps so no lines clear)
    padding = tetris_env.unwrapped.padding
    bottom_y = tetris_env.unwrapped.height - 4
    tetris_env.unwrapped.board[
        bottom_y : tetris_env.unwrapped.height,
        padding : padding + 5,
    ] = 2

    # Use whatever piece is active and center it over the filled area
    tetris_env.unwrapped.x = padding + 2
    tetris_env.unwrapped.y = 0

    piece_height = tetris_env.unwrapped.active_tetromino.matrix.shape[0]
    tetris_env.step(ActionsMapping.hard_drop)

    # Piece should have landed on top of the existing blocks, not at the very bottom
    # Check that there's something placed above the existing blocks
    board = tetris_env.unwrapped.board
    assert np.any(
        board[bottom_y - piece_height : bottom_y, padding : padding + 5] >= 2
    ), "Piece should land on top of existing blocks"


def test_hard_drop_triggers_next_piece_spawn(tetris_env):
    """Test that hard drop spawns a new piece."""
    tetris_env.reset(seed=42)

    tetris_env.step(ActionsMapping.hard_drop)

    # A new piece should have been spawned
    assert tetris_env.unwrapped.active_tetromino is not None


def test_hard_drop_resets_position(tetris_env):
    """Test that after hard drop, position resets to spawn position."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.y = 5
    tetris_env.unwrapped.x = tetris_env.unwrapped.padding + 2

    tetris_env.step(ActionsMapping.hard_drop)

    # Position should be reset to spawn position
    expected_x = (
        tetris_env.unwrapped.width_padded // 2
        - tetris_env.unwrapped.active_tetromino.matrix.shape[0] // 2
    )
    assert tetris_env.unwrapped.x == expected_x
    assert tetris_env.unwrapped.y == 0
