"""Integration tests: full gameplay simulation."""

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


class TestIntegration:
    def test_random_play_reaches_game_over(self):
        key = jax.random.PRNGKey(0)
        key, state, _ = reset(TETROMINOES, key, CONFIG)
        for i in range(10000):
            key, subkey = jax.random.split(key)
            action = jax.random.randint(subkey, (), 0, 7)
            state, _, _, terminated, _ = step(TETROMINOES, state, int(action), CONFIG)
            if bool(terminated):
                break
        assert bool(state.game_over)

    def test_score_non_decreasing(self):
        key = jax.random.PRNGKey(1)
        key, state, _ = reset(TETROMINOES, key, CONFIG)
        prev_score = float(state.score)
        for i in range(500):
            key, subkey = jax.random.split(key)
            action = jax.random.randint(subkey, (), 0, 7)
            state, _, _, terminated, _ = step(TETROMINOES, state, int(action), CONFIG)
            curr_score = float(state.score)
            assert curr_score >= prev_score
            prev_score = curr_score
            if bool(terminated):
                break

    def test_reproducibility(self):
        key = jax.random.PRNGKey(42)
        actions_key = jax.random.PRNGKey(99)

        # Generate action sequence
        action_keys = jax.random.split(actions_key, 200)
        actions = [int(jax.random.randint(k, (), 0, 7)) for k in action_keys]

        # Run 1
        _, s1, _ = reset(TETROMINOES, key, CONFIG)
        for a in actions:
            s1, _, _, t, _ = step(TETROMINOES, s1, a, CONFIG)
            if bool(t):
                break

        # Run 2
        _, s2, _ = reset(TETROMINOES, key, CONFIG)
        for a in actions:
            s2, _, _, t, _ = step(TETROMINOES, s2, a, CONFIG)
            if bool(t):
                break

        assert jnp.array_equal(s1.board, s2.board)
        assert float(s1.score) == float(s2.score)
        assert bool(s1.game_over) == bool(s2.game_over)
