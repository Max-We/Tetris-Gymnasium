import jax
import jax.numpy as jnp
from jax import random
import pytest

from tetris_gymnasium.envs.tetris_fn import reset, step
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.tetrominoes import TETROMINOES


@pytest.fixture
def env_config():
    return EnvConfig(width=10, height=20, padding=4, queue_size=7)


def test_jit_random_actions_until_termination(env_config):
    """Test stepping through the environment with random actions using JIT until termination.

    Also makes sure, that the env is JIT-able.
    """

    reset_static = jax.jit(reset, static_argnames=['config'])
    step_static = jax.jit(step, static_argnames=['config'])

    @jax.jit
    def run_episode(key):
        key, reset_key = random.split(key)
        key, initial_state = reset_static(TETROMINOES, reset_key, config=env_config)

        def body_fun(carry):
            key, state, done = carry
            key, step_key = random.split(key)
            action = random.randint(step_key, (), 0, 7)  # Assuming 7 possible actions (0-6)
            key, new_state, reward, done, info = step_static(TETROMINOES, key, state, action, config=env_config)
            return (key, new_state, done)

        def cond_fun(carry):
            _, _, done = carry
            return ~done

        final_carry = jax.lax.while_loop(cond_fun, body_fun, (key, initial_state, False))
        return final_carry

    # Run the jitted function
    init_key = random.PRNGKey(0)
    final_key, final_state, terminated = jax.jit(run_episode)(init_key)
    final_key.block_until_ready()

    # Assert that the game terminated
    assert terminated

# Run the test
if __name__ == "__main__":
    pytest.main([__file__])