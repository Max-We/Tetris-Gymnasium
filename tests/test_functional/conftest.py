"""Shared fixtures for functional Tetris tests."""

import jax
import pytest

from tetris_gymnasium.envs.tetris_fn import reset
from tetris_gymnasium.functional.core import EnvConfig, create_board
from tetris_gymnasium.functional.tetrominoes import TETROMINOES


@pytest.fixture
def tetrominoes():
    """Return the TETROMINOES object."""
    return TETROMINOES


@pytest.fixture
def default_config():
    """Return default env config."""
    return EnvConfig(
        width=10,
        height=20,
        padding=4,
        queue_size=7,
        gravity_enabled=True,
    )


@pytest.fixture
def no_gravity_config():
    """Return config with gravity disabled."""
    return EnvConfig(
        width=10,
        height=20,
        padding=4,
        queue_size=7,
        gravity_enabled=False,
    )


@pytest.fixture
def rng_key():
    """Return a fixed JAX PRNG key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def initial_state(tetrominoes, rng_key, default_config):
    """Return initial state from reset with default config."""
    _key, state, _obs = reset(
        tetrominoes,
        rng_key,
        default_config,
    )
    return state


@pytest.fixture
def initial_state_no_gravity(tetrominoes, rng_key, no_gravity_config):
    """Return initial state from reset with no gravity."""
    _key, state, _obs = reset(
        tetrominoes,
        rng_key,
        no_gravity_config,
    )
    return state


@pytest.fixture
def empty_board(default_config, tetrominoes):
    """Return an empty board with default config."""
    return create_board(default_config, tetrominoes)
