"""Tests for movement actions (0=left, 1=right, 2=soft_drop)."""

import jax

from tetris_gymnasium.envs.tetris_fn import reset, step


class TestMovement:
    def test_move_left(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        orig_x = int(state.x)
        new_state, _, _, _, _ = step(tetrominoes, state, 0, no_gravity_config)
        assert int(new_state.x) == orig_x - 1

    def test_move_right(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        orig_x = int(state.x)
        new_state, _, _, _, _ = step(tetrominoes, state, 1, no_gravity_config)
        assert int(new_state.x) == orig_x + 1

    def test_soft_drop(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        orig_y = int(state.y)
        new_state, _, _, _, _ = step(tetrominoes, state, 2, no_gravity_config)
        assert int(new_state.y) == orig_y + 1

    def test_blocked_left_at_border(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        # Move left many times until blocked
        for _ in range(20):
            state, _, _, _, _ = step(tetrominoes, state, 0, no_gravity_config)
        prev_x = int(state.x)
        state, _, _, _, _ = step(tetrominoes, state, 0, no_gravity_config)
        assert int(state.x) == prev_x  # blocked

    def test_blocked_right_at_border(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        for _ in range(20):
            state, _, _, _, _ = step(tetrominoes, state, 1, no_gravity_config)
        prev_x = int(state.x)
        state, _, _, _, _ = step(tetrominoes, state, 1, no_gravity_config)
        assert int(state.x) == prev_x

    def test_soft_drop_blocked_at_bottom(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        for _ in range(30):
            state, _, _, _, _ = step(tetrominoes, state, 2, no_gravity_config)
        prev_y = int(state.y)
        state, _, _, _, _ = step(tetrominoes, state, 2, no_gravity_config)
        assert int(state.y) == prev_y

    def test_multiple_consecutive_moves(self, tetrominoes, rng_key, no_gravity_config):
        _, state, _ = reset(tetrominoes, rng_key, no_gravity_config)
        orig_x = int(state.x)
        # Move right 3 times
        for _ in range(3):
            state, _, _, _, _ = step(tetrominoes, state, 1, no_gravity_config)
        assert int(state.x) == orig_x + 3

    def test_blocked_by_placed_piece(self, tetrominoes, no_gravity_config):
        # Hard drop to place a piece, then check collision
        key = jax.random.PRNGKey(123)
        _, state, _ = reset(tetrominoes, key, no_gravity_config)
        # Hard drop to place piece
        state, _, _, _, _ = step(tetrominoes, state, 6, no_gravity_config)
        # Soft drop the new piece toward the placed one
        for _ in range(30):
            prev_y = int(state.y)
            state, _, _, terminated, _ = step(tetrominoes, state, 2, no_gravity_config)
            if bool(terminated) or int(state.y) == prev_y:
                break
        # Should eventually be blocked
        assert int(state.y) == prev_y or bool(terminated)
