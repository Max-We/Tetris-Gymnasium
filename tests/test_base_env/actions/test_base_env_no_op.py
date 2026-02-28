from tetris_gymnasium.mappings.actions import ActionsMapping


def test_no_op_does_not_change_x(tetris_env):
    """Test that no-op action doesn't change the x position."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.x = 7
    original_x = tetris_env.unwrapped.x

    tetris_env.step(ActionsMapping.no_op)

    assert tetris_env.unwrapped.x == original_x


def test_no_op_with_gravity_moves_piece_down(tetris_env):
    """Test that no-op with gravity still moves the piece down."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.y = 0

    tetris_env.step(ActionsMapping.no_op)

    assert tetris_env.unwrapped.y == 1, "Gravity should move piece down on no-op"


def test_no_op_without_gravity_does_not_move(tetris_env_no_gravity):
    """Test that no-op without gravity doesn't move the piece at all."""
    tetris_env_no_gravity.reset(seed=42)
    original_x = tetris_env_no_gravity.unwrapped.x
    original_y = tetris_env_no_gravity.unwrapped.y

    tetris_env_no_gravity.step(ActionsMapping.no_op)

    assert tetris_env_no_gravity.unwrapped.x == original_x
    assert tetris_env_no_gravity.unwrapped.y == original_y
