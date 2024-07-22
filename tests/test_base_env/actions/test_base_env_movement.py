from tetris_gymnasium.mappings.actions import ActionsMapping


# Free movement tests: left, right, down
def test_move_right(tetris_env):
    """Test moving a tetromino to the right in a regular scenario."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.x = 5
    tetris_env.step(ActionsMapping.move_right)
    assert tetris_env.unwrapped.x == 6


def test_move_left(tetris_env):
    """Test moving a tetromino to the left in a regular scenario."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.x = 5
    tetris_env.step(ActionsMapping.move_left)
    assert tetris_env.unwrapped.x == 4


def test_move_down(tetris_env):
    """Test moving a tetromino down in a regular scenario."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.y = 5
    tetris_env.step(ActionsMapping.move_down)
    assert tetris_env.unwrapped.y == 7  # movement down (1) + gravity (1)


# Blocked movement by board edges tests: left, right, down
def test_move_right_at_border(tetris_env):
    """Test moving a tetromino to the right when it's at the right border."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.x = (
        tetris_env.unwrapped.width + tetris_env.unwrapped.padding - 1
    )
    tetris_env.step(ActionsMapping.move_right)
    assert (
        tetris_env.unwrapped.x
        == tetris_env.unwrapped.width + tetris_env.unwrapped.padding - 1
    )


def test_move_left_at_border(tetris_env):
    """Test moving a tetromino to the left when it's at the left border."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.x = tetris_env.unwrapped.padding
    tetris_env.step(ActionsMapping.move_left)
    assert tetris_env.unwrapped.x == tetris_env.unwrapped.padding


def test_move_down_at_border(tetris_env):
    """Test moving a tetromino down when it's at the bottom border."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.y = tetris_env.unwrapped.height - 1
    tetris_env.step(ActionsMapping.move_down)
    assert tetris_env.unwrapped.y == tetris_env.unwrapped.height - 1


# Blocked movement by other tetrominos tests: left, right, down
def test_move_right_blocked_by_tetromino(tetris_env):
    """Test moving a tetromino to the right when blocked by another tetromino."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.x = 5
    tetris_env.unwrapped.board[
        0 : tetris_env.unwrapped.height,
        tetris_env.unwrapped.x + 1 : tetris_env.unwrapped.x + 5,
    ] = 2
    tetris_env.step(ActionsMapping.move_right)
    assert tetris_env.unwrapped.x == 5


def test_move_left_blocked_by_tetromino(tetris_env):
    """Test moving a tetromino to the left when blocked by another tetromino."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.x = 5
    tetris_env.unwrapped.board[
        0 : tetris_env.unwrapped.height,
        tetris_env.unwrapped.x - 4 : tetris_env.unwrapped.x,
    ] = 2
    tetris_env.step(ActionsMapping.move_left)
    assert tetris_env.unwrapped.x == 5


def test_move_down_blocked_by_tetromino(tetris_env):
    """Test moving a tetromino down when blocked by another tetromino."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.y = 5
    tetris_env.unwrapped.board[
        tetris_env.unwrapped.y + 1 : tetris_env.unwrapped.y + 5,
        0 : tetris_env.unwrapped.width,
    ] = 2
    tetris_env.step(ActionsMapping.move_down)
    assert tetris_env.unwrapped.y == 5


# Gravity test
def test_gravity(tetris_env):
    """Test gravity effect on a tetromino."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.y = 0
    tetris_env.step(ActionsMapping.no_op)
    assert tetris_env.unwrapped.y == 1
