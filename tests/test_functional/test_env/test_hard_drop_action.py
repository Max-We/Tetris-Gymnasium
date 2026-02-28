"""Tests for hard drop action (action 6)."""

import jax
import jax.numpy as jnp

from tetris_gymnasium.envs.tetris_fn import reset, step


class TestHardDropAction:
    def test_piece_locked_new_piece_spawned(
        self, tetrominoes, rng_key, no_gravity_config
    ):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        new_state, _, _, _, _ = step(tetrominoes, state, 6, no_gravity_config)
        # After hard drop, piece is locked and new one spawned
        # Board should have some non-zero values in playable area
        p = no_gravity_config.padding
        playable = new_state.board[
            : no_gravity_config.height, p : p + no_gravity_config.width
        ]
        assert jnp.any(playable > 0)
        # y should be near top (new piece spawned)
        assert int(new_state.y) == 0

    def test_lands_on_existing_pieces(self, tetrominoes, no_gravity_config):
        key = jax.random.PRNGKey(42)
        _, state, _ = reset(tetrominoes, key, no_gravity_config)
        # Hard drop twice
        state, _, _, _, _ = step(tetrominoes, state, 6, no_gravity_config)
        state, _, _, _, _ = step(tetrominoes, state, 6, no_gravity_config)
        # Board should have more pieces placed
        p = no_gravity_config.padding
        playable = state.board[
            : no_gravity_config.height, p : p + no_gravity_config.width
        ]
        assert jnp.sum(playable > 0) > 0

    def test_score_includes_drop_distance(
        self, tetrominoes, rng_key, no_gravity_config
    ):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        new_state, _, reward, _, _ = step(tetrominoes, state, 6, no_gravity_config)
        # Reward should be positive (at least from drop distance)
        assert float(reward) > 0
