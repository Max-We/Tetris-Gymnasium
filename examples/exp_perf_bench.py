import time
import jax
import jax.numpy as jnp
from jax import random
import chex
from typing import Tuple
import gymnasium as gym

from examples.exp_t import TETRIS_CONSTANTS, TetrisConstants
from tetris_gymnasium.envs.tetris_fn import TetrisState, TetrisConfig, reset, step
from tetris_gymnasium.envs.tetris import Tetris

# JAX Environment Setup
CONFIG = TetrisConfig(width=10, height=20, padding=10, queue_size=3)

# reset_static = jax.jit(reset, static_argnames=['config'])
# step_static = jax.jit(step, static_argnames=['config'])


# @jax.jit
def play_episode_jax(key: chex.PRNGKey, num_steps: int) -> Tuple[chex.PRNGKey, TetrisState, int]:
    key, subkey = random.split(key)
    key, state = reset(TETRIS_CONSTANTS, subkey, CONFIG)

    def body_fun(carry):
        key, state, step_count = carry
        key, subkey = random.split(key)
        action = random.randint(subkey, (), 0, 7)  # 7 possible actions
        key, new_state, reward, game_over, info = step(TETRIS_CONSTANTS, key, state, action, CONFIG)

        # Reset if game over
        key, reset_key = random.split(key)
        # reset_state = reset_static(TETRIS_CONSTANTS, reset_key, CONFIG)[1]
        state = jax.lax.cond(
            game_over,
            lambda _: reset(TETRIS_CONSTANTS, reset_key, CONFIG)[1],
            lambda _: new_state,
            None
        )

        return key, state, step_count + 1

    def cond_fun(carry):
        key, state, step_count = carry
        return step_count < num_steps

    key, final_state, steps_taken = jax.lax.while_loop(cond_fun, body_fun, (key, state, 0))
    return key, final_state, steps_taken


# @jax.jit
def run_jax_environment(key: chex.PRNGKey, num_steps: int) -> Tuple[chex.PRNGKey, TetrisState, int]:
    return play_episode_jax(key, num_steps)


# Gym Environment Setup
def run_gym_environment(num_steps: int) -> float:
    env = gym.make("tetris_gymnasium/Tetris")
    env.reset(seed=42)

    start_time = time.time()
    for _ in range(num_steps):
        action = env.action_space.sample()
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            env.reset()

    end_time = time.time()
    env.close()
    return (end_time - start_time) / num_steps


def main():
    num_steps = 5000

    print("Running JAX environment...")
    key = random.PRNGKey(0)

    # Compile the function (this may take a moment)
    key, final_state, steps_taken = run_jax_environment(key, 10) # Warm-up run
    key = key.block_until_ready()

    # Measure the actual run
    start_time = time.time()
    key, final_state, steps_taken = run_jax_environment(key, num_steps)
    key = key.block_until_ready()
    end_time = time.time()

    jax_time_per_step = (end_time - start_time) / num_steps
    print(f"JAX environment: {jax_time_per_step:.6f} seconds per step")

    print("\nRunning Gym environment...")
    gym_time_per_step = run_gym_environment(num_steps)
    print(f"Gym environment: {gym_time_per_step:.6f} seconds per step")

    speedup = gym_time_per_step / jax_time_per_step
    print(f"\nSpeedup factor: {speedup:.2f}x")


if __name__ == "__main__":
    jax.config.update("jax_disable_jit", True)
    main()
