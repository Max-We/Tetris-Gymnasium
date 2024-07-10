import gymnasium as gym
import numpy as np
import pytest

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_gymnasium.mappings.rewards import RewardsMapping

@pytest.fixture
def tetris_env():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env.reset(seed=42)
    yield env
    env.close()

@pytest.fixture
def o_tetromino():
    """Fixture to create and return an O-tetromino."""
    return Tetris.TETROMINOES[1]

def test_game_over_stack_too_high(tetris_env, o_tetromino):
    """Test game over condition when stack is too high after hard drop."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.board[2:tetris_env.unwrapped.height, tetris_env.unwrapped.padding:-(tetris_env.unwrapped.padding+1)] = 2
    tetris_env.unwrapped.active_tetromino = o_tetromino
    tetris_env.unwrapped.x = tetris_env.unwrapped.width_padded // 2 - tetris_env.unwrapped.active_tetromino.matrix.shape[0] // 2
    tetris_env.unwrapped.y = 0

    observation, reward, terminated, truncated, info = tetris_env.step(ActionsMapping.hard_drop)

    assert terminated
    assert reward == RewardsMapping.game_over

def test_game_over_invalid_position(tetris_env, o_tetromino):
    """Test game over condition when active tetromino is inside other tetrominos or the wall."""
    tetris_env.reset(seed=42)
    tetris_env.unwrapped.board[0:tetris_env.unwrapped.height, tetris_env.unwrapped.padding:-(tetris_env.unwrapped.padding+1)] = 2
    tetris_env.unwrapped.active_tetromino = o_tetromino
    tetris_env.unwrapped.x = tetris_env.unwrapped.width_padded // 2 - tetris_env.unwrapped.active_tetromino.matrix.shape[0] // 2
    tetris_env.unwrapped.y = 0

    observation, reward, terminated, truncated, info = tetris_env.step(ActionsMapping.hard_drop)

    assert terminated
    assert reward == RewardsMapping.game_over
