"""Tests for step() general behavior."""

import jax.numpy as jnp

from tetris_gymnasium.envs.tetris_fn import reset, step


class TestStep:
    def test_returns_correct_tuple(self, tetrominoes, rng_key, default_config):
        _, state, _ = reset(tetrominoes, rng_key, default_config)
        result = step(tetrominoes, state, 5, default_config)
        assert len(result) == 5
        new_state, obs, reward, terminated, info = result
        assert "lines_cleared" in info

    def test_no_op_when_game_over(self, tetrominoes, rng_key, default_config):
        _, state, _ = reset(tetrominoes, rng_key, default_config)
        # Force game over
        state = state.replace(game_over=jnp.bool_(True))
        new_state, _, reward, terminated, _ = step(
            tetrominoes, state, 0, default_config
        )
        assert bool(terminated)
        assert float(reward) == 0.0
        assert jnp.array_equal(new_state.board, state.board)

    def test_reward_equals_score_diff(self, tetrominoes, rng_key, default_config):
        _, state, _ = reset(tetrominoes, rng_key, default_config)
        new_state, _, reward, _, _ = step(tetrominoes, state, 5, default_config)
        assert float(reward) == (float(new_state.score) - float(state.score))

    def test_observation_shape(self, tetrominoes, rng_key, default_config):
        _, state, _ = reset(tetrominoes, rng_key, default_config)
        _, obs, _, _, _ = step(tetrominoes, state, 5, default_config)
        assert obs.shape == (20, 10)
