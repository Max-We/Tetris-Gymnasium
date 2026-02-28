"""Fixtures for the Tetris environment and tetrominoes."""

import copy

import gymnasium as gym
import numpy as np
import pytest

from tetris_gymnasium.envs import Tetris


@pytest.fixture
def tetris_env():
    """Fixture to create and return a Tetris environment."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env.reset(seed=42)
    yield env
    env.close()


# Tetrominoes
@pytest.fixture
def o_tetromino():
    """Fixture to create and return an O-tetromino."""
    return copy.deepcopy(Tetris.TETROMINOES[1])


@pytest.fixture
def tetris_env_no_gravity():
    """Fixture to create and return a Tetris environment with gravity disabled."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi", gravity=False)
    env.reset(seed=42)
    yield env
    env.close()


@pytest.fixture
def vertical_i_tetromino():
    """Fixture to create and return a vertical I-tetromino."""
    tetromino = copy.deepcopy(Tetris.TETROMINOES[0])
    tetromino.matrix = np.rot90(tetromino.matrix)
    return tetromino
