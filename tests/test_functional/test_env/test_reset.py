"""Tests for reset()."""

import jax
import jax.numpy as jnp

from tetris_gymnasium.envs.tetris_fn import reset


class TestReset:
    def test_returns_tuple(self, tetrominoes, rng_key, default_config):
        result = reset(tetrominoes, rng_key, default_config)
        assert len(result) == 3  # (key, state, observation)

    def test_state_initial_values(self, initial_state):
        assert int(initial_state.rotation) == 0
        assert not bool(initial_state.game_over)
        assert float(initial_state.score) == 0.0

    def test_board_playable_area_empty(self, initial_state, default_config):
        p = default_config.padding
        playable = initial_state.board[
            : default_config.height, p : p + default_config.width
        ]
        assert jnp.all(playable == 0)

    def test_observation_shape(self, tetrominoes, rng_key, default_config):
        _, _, obs = reset(tetrominoes, rng_key, default_config)
        assert obs.shape == (20, 10)

    def test_deterministic_same_key(self, tetrominoes, default_config):
        key = jax.random.PRNGKey(42)
        _, s1, o1 = reset(tetrominoes, key, default_config)
        _, s2, o2 = reset(tetrominoes, key, default_config)
        assert jnp.array_equal(s1.board, s2.board)
        assert jnp.array_equal(o1, o2)
        assert int(s1.active_tetromino) == int(s2.active_tetromino)

    def test_different_keys_different_states(self, tetrominoes, default_config):
        _, s1, _ = reset(tetrominoes, jax.random.PRNGKey(0), default_config)
        _, s2, _ = reset(tetrominoes, jax.random.PRNGKey(999), default_config)
        # Very likely different active tetromino or queue
        different = int(s1.active_tetromino) != int(
            s2.active_tetromino
        ) or not jnp.array_equal(s1.queue, s2.queue)
        assert different
