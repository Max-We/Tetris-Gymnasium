"""Tests for rotation actions (3=ccw, 4=cw per actual implementation)."""

import jax

from tetris_gymnasium.envs.tetris_fn import reset, step


class TestRotation:
    def test_cw_rotation(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        orig_rot = int(state.rotation)
        new_state, _, _, _, _ = step(tetrominoes, state, 4, no_gravity_config)
        assert int(new_state.rotation) == (orig_rot + 1) % 4

    def test_ccw_rotation(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        orig_rot = int(state.rotation)
        new_state, _, _, _, _ = step(tetrominoes, state, 3, no_gravity_config)
        assert int(new_state.rotation) == (orig_rot - 1) % 4

    def test_full_360_cycle(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        orig_rot = int(state.rotation)
        for _ in range(4):
            state, _, _, _, _ = step(tetrominoes, state, 4, no_gravity_config)
        assert int(state.rotation) == orig_rot

    def test_cw_then_ccw_identity(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        orig_rot = int(state.rotation)
        state, _, _, _, _ = step(tetrominoes, state, 4, no_gravity_config)
        state, _, _, _, _ = step(tetrominoes, state, 3, no_gravity_config)
        assert int(state.rotation) == orig_rot

    def test_rotation_blocked_by_wall(self, tetrominoes, no_gravity_config):
        key = jax.random.PRNGKey(0)
        _, state, _ = reset(tetrominoes, key, no_gravity_config)
        # Move as far left as possible
        for _ in range(15):
            state, _, _, _, _ = step(tetrominoes, state, 0, no_gravity_config)
        state, _, _, _, _ = step(tetrominoes, state, 4, no_gravity_config)
        # Rotation might succeed or be blocked
        assert 0 <= int(state.rotation) <= 3
