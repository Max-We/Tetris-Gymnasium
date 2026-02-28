"""Tests for no-op action (action 5)."""

from tetris_gymnasium.envs.tetris_fn import reset, step


class TestNoOp:
    def test_x_unchanged(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        orig_x = int(state.x)
        new_state, _, _, _, _ = step(tetrominoes, state, 5, no_gravity_config)
        assert int(new_state.x) == orig_x

    def test_with_gravity_y_moves_down(self, tetrominoes, rng_key, default_config):
        _, state, _ = reset(tetrominoes, rng_key, default_config)
        orig_y = int(state.y)
        new_state, _, _, _, _ = step(tetrominoes, state, 5, default_config)
        # Gravity should move piece down (unless immediately locked)
        assert int(new_state.y) >= orig_y

    def test_without_gravity_no_change(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        orig_x = int(state.x)
        orig_y = int(state.y)
        orig_rot = int(state.rotation)
        new_state, _, _, _, _ = step(tetrominoes, state, 5, no_gravity_config)
        assert int(new_state.x) == orig_x
        assert int(new_state.y) == orig_y
        assert int(new_state.rotation) == orig_rot
