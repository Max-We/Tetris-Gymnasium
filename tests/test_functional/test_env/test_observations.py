"""Tests for get_observation and get_feature_observation."""

import jax.numpy as jnp

from tetris_gymnasium.envs.tetris_fn import get_observation, reset


class TestObservations:
    def test_observation_shape(self, tetrominoes, rng_key, default_config):
        _, state, obs = reset(tetrominoes, rng_key, default_config)
        assert obs.shape == (20, 10)

    def test_observation_values(self, tetrominoes, rng_key, default_config):
        _, state, _ = reset(tetrominoes, rng_key, default_config)
        obs = get_observation(
            state.board,
            state.x,
            state.y,
            state.active_tetromino,
            state.rotation,
            state.game_over,
            tetrominoes,
            default_config,
        )
        # Board values: 0 (empty), 1 (placed), -1 (active piece)
        unique = set(obs.flatten().tolist())
        assert unique.issubset({-1, 0, 1})

    def test_padding_stripped(self, tetrominoes, rng_key, default_config):
        _, state, _ = reset(tetrominoes, rng_key, default_config)
        obs = get_observation(
            state.board,
            state.x,
            state.y,
            state.active_tetromino,
            state.rotation,
            state.game_over,
            tetrominoes,
            default_config,
        )
        # Observation should NOT contain bedrock
        # (value 1 from padding is stripped)
        # The returned shape should be playable area only
        assert obs.shape == (default_config.height, default_config.width)

    def test_game_over_no_active_piece(self, tetrominoes, rng_key, default_config):
        _, state, _ = reset(tetrominoes, rng_key, default_config)
        state = state.replace(game_over=jnp.bool_(True))
        obs = get_observation(
            state.board,
            state.x,
            state.y,
            state.active_tetromino,
            state.rotation,
            state.game_over,
            tetrominoes,
            default_config,
        )
        # No -1 values when game is over
        assert not jnp.any(obs == -1)
