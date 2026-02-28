"""Tests for batched_reset and batched_step."""

import jax
import jax.numpy as jnp

from tetris_gymnasium.envs.tetris_fn import batched_reset, batched_step, step
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

CONFIG = EnvConfig(
    width=10,
    height=20,
    padding=4,
    queue_size=7,
    gravity_enabled=True,
)
BATCH_SIZE = 4


class TestBatchedReset:
    def test_leading_dim(self):
        keys = jax.random.split(jax.random.PRNGKey(0), BATCH_SIZE)
        _, states, obs = batched_reset(
            TETROMINOES, keys, config=CONFIG, batch_size=BATCH_SIZE
        )
        assert states.board.shape[0] == BATCH_SIZE
        assert states.x.shape[0] == BATCH_SIZE
        assert obs.shape[0] == BATCH_SIZE

    def test_different_keys_different_states(self):
        keys = jax.random.split(jax.random.PRNGKey(0), BATCH_SIZE)
        _, states, _ = batched_reset(
            TETROMINOES, keys, config=CONFIG, batch_size=BATCH_SIZE
        )
        # Just check it doesn't crash and shapes are right
        assert states.active_tetromino.shape[0] == BATCH_SIZE

    def test_same_keys_same_states(self):
        key = jax.random.PRNGKey(42)
        keys = jnp.stack([key] * BATCH_SIZE)
        _, states, _ = batched_reset(
            TETROMINOES, keys, config=CONFIG, batch_size=BATCH_SIZE
        )
        for i in range(1, BATCH_SIZE):
            assert jnp.array_equal(states.board[0], states.board[i])
            at0 = int(states.active_tetromino[0])
            ati = int(states.active_tetromino[i])
            assert at0 == ati


class TestBatchedStep:
    def test_processes_all_envs(self):
        keys = jax.random.split(jax.random.PRNGKey(0), BATCH_SIZE)
        _, states, _ = batched_reset(
            TETROMINOES, keys, config=CONFIG, batch_size=BATCH_SIZE
        )
        actions = jnp.array([0, 1, 5, 5])  # different actions per env
        new_states, obs, rewards, terminated, info = batched_step(
            TETROMINOES, states, actions, config=CONFIG
        )
        assert new_states.board.shape[0] == BATCH_SIZE
        assert obs.shape == (BATCH_SIZE, 20, 10)

    def test_matches_sequential(self):
        keys = jax.random.split(jax.random.PRNGKey(0), BATCH_SIZE)
        _, states, _ = batched_reset(
            TETROMINOES, keys, config=CONFIG, batch_size=BATCH_SIZE
        )
        actions = jnp.array([0, 1, 2, 5])

        # Batched
        b_states, b_obs, b_rew, b_term, b_info = batched_step(
            TETROMINOES, states, actions, config=CONFIG
        )

        # Sequential
        for i in range(BATCH_SIZE):
            single_state_fields = {
                field: getattr(states, field)[i]
                for field in states.__dataclass_fields__
            }
            from tetris_gymnasium.functional.core import State

            single_state = State(**single_state_fields)
            s_state, s_obs, s_rew, s_term, s_info = step(
                TETROMINOES, single_state, int(actions[i]), CONFIG
            )
            assert jnp.allclose(b_obs[i], s_obs)
            assert jnp.allclose(b_rew[i], s_rew)
