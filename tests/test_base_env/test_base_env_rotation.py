import copy
import gymnasium as gym
import numpy as np
import pytest

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.actions import ActionsMapping


@pytest.fixture
def tetris_env():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env.reset(seed=42)
    yield env
    env.close()


# Free rotation tests: clockwise, counter-clockwise
def test_rotate_clockwise(tetris_env):
    """Test rotating a tetromino clockwise in a regular scenario."""
    tetris_env.reset(seed=42)
    tetromino = np.copy(tetris_env.unwrapped.active_tetromino.matrix)
    tetris_env.step(ActionsMapping.rotate_clockwise)
    assert np.all(np.rot90(tetromino) == tetris_env.unwrapped.active_tetromino.matrix)


def test_rotate_counter_clockwise(tetris_env):
    """Test rotating a tetromino counter-clockwise in a regular scenario."""
    tetris_env.reset(seed=42)
    tetromino = np.copy(tetris_env.unwrapped.active_tetromino.matrix)
    tetris_env.step(ActionsMapping.rotate_counterclockwise)
    assert np.array_equal(np.rot90(tetromino, -1), tetris_env.unwrapped.active_tetromino.matrix)


# Test rotation blocked by other tetrominos
def test_rotate_clockwise_blocked(tetris_env):
    """Test rotating a tetromino clockwise when blocked by another tetromino."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.board[0:tetris_env.unwrapped.height, tetris_env.unwrapped.x:tetris_env.unwrapped.x + 4] = 2
    tetromino = copy.copy(tetris_env.unwrapped.active_tetromino)
    tetris_env.step(ActionsMapping.rotate_clockwise)
    assert np.array_equal(tetromino.matrix, tetris_env.unwrapped.active_tetromino.matrix)


def test_rotate_counter_clockwise_blocked(tetris_env):
    """Test rotating a tetromino counter-clockwise when blocked by another tetromino."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.board[0:tetris_env.unwrapped.height, tetris_env.unwrapped.x:tetris_env.unwrapped.x + 4] = 2
    tetromino = copy.copy(tetris_env.unwrapped.active_tetromino)
    tetris_env.step(ActionsMapping.rotate_counterclockwise)
    assert np.array_equal(tetromino.matrix, tetris_env.unwrapped.active_tetromino.matrix)
