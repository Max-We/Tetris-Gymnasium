"""Tests for JAX compatibility: JIT, determinism, vmap."""

from functools import partial

import jax
import jax.numpy as jnp

from tetris_gymnasium.envs.tetris_fn import reset, step
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

CONFIG = EnvConfig(
    width=10,
    height=20,
    padding=4,
    queue_size=7,
    gravity_enabled=True,
)


class TestJITCompatibility:
    def test_jit_reset(self):
        key = jax.random.PRNGKey(42)
        reset_fn = partial(reset, TETROMINOES, config=CONFIG)
        _, s1, o1 = reset_fn(key)
        _, s2, o2 = jax.jit(reset_fn)(key)
        assert jnp.array_equal(s1.board, s2.board)
        assert jnp.array_equal(o1, o2)

    def test_jit_step(self):
        key = jax.random.PRNGKey(42)
        _, state, _ = reset(TETROMINOES, key, CONFIG)
        step_fn = partial(step, TETROMINOES, config=CONFIG)
        s1, o1, r1, t1, i1 = step_fn(state, 5)
        s2, o2, r2, t2, i2 = jax.jit(step_fn)(state, 5)
        assert jnp.array_equal(o1, o2)
        assert float(r1) == float(r2)

    def test_core_functions_jit(self):
        from tetris_gymnasium.functional.core import collision, create_board, score
        from tetris_gymnasium.functional.tetrominoes import get_tetromino_matrix

        board = create_board(CONFIG, TETROMINOES)
        mat = get_tetromino_matrix(TETROMINOES, 0, 0)

        # collision
        r1 = collision(board, mat, 7, 0)
        r2 = jax.jit(collision)(board, mat, 7, 0)
        assert bool(r1) == bool(r2)

        # score
        assert int(jax.jit(partial(score, CONFIG))(2)) == int(score(CONFIG, 2))


class TestDeterminism:
    def test_identical_states_across_resets(self):
        key = jax.random.PRNGKey(42)
        _, s1, _ = reset(TETROMINOES, key, CONFIG)
        _, s2, _ = reset(TETROMINOES, key, CONFIG)
        assert jnp.array_equal(s1.board, s2.board)
        assert int(s1.active_tetromino) == int(s2.active_tetromino)
        assert int(s1.x) == int(s2.x)
        assert int(s1.y) == int(s2.y)

    def test_identical_step_sequences(self):
        key = jax.random.PRNGKey(42)
        actions = [0, 1, 4, 2, 5, 6, 3]
        _, s1, _ = reset(TETROMINOES, key, CONFIG)
        _, s2, _ = reset(TETROMINOES, key, CONFIG)
        for a in actions:
            s1, _, _, _, _ = step(TETROMINOES, s1, a, CONFIG)
            s2, _, _, _, _ = step(TETROMINOES, s2, a, CONFIG)
        assert jnp.array_equal(s1.board, s2.board)
        assert float(s1.score) == float(s2.score)


class TestVmap:
    def test_vmap_step_matches_loop(self):
        batch = 4
        keys = jax.random.split(jax.random.PRNGKey(0), batch)
        # Reset each independently
        states_list = []
        for k in keys:
            _, s, _ = reset(TETROMINOES, k, CONFIG)
            states_list.append(s)

        # Stack into batched state
        from tetris_gymnasium.functional.core import State

        batched = State(
            **{
                field: jnp.stack([getattr(s, field) for s in states_list])
                for field in State.__dataclass_fields__
            }
        )

        actions = jnp.array([0, 1, 2, 5])
        step_fn = partial(step, TETROMINOES, config=CONFIG)
        vmapped = jax.vmap(step_fn, in_axes=(0, 0))
        b_states, b_obs, b_rew, b_term, b_info = vmapped(batched, actions)

        for i in range(batch):
            s_state, s_obs, s_rew, _, _ = step_fn(states_list[i], int(actions[i]))
            assert jnp.allclose(b_obs[i], s_obs)
