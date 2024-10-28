import os
import time
import jax
import jax.numpy as jnp
from jax import random
import chex
from typing import Tuple
import gymnasium as gym

from tetris_gymnasium.envs.tetris_fn import State, EnvConfig, batched_reset, batched_step
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

# JAX Environment Setup
CONFIG = EnvConfig(width=10, height=20, padding=10, queue_size=3)
BATCH_SIZE = 128  # We can experiment with different batch sizes


@jax.jit
def play_parallel_episodes_jax(key: chex.PRNGKey, num_steps: int) -> Tuple[chex.PRNGKey, chex.Array, float]:
    key, reset_key = random.split(key)
    reset_keys = random.split(reset_key, BATCH_SIZE)

    # Reset all environments in parallel
    keys, states, _ = batched_reset(
        TETROMINOES,
        reset_keys,
        config=CONFIG,
        batch_size=BATCH_SIZE
    )

    # Initialize total steps counter
    total_steps = jnp.zeros((), dtype=jnp.int32)

    def body_fun(carry):
        keys, states, total_steps, rng = carry

        # Generate random actions
        rng, subkey = random.split(rng)
        action_keys = random.split(subkey, BATCH_SIZE)
        actions = jax.vmap(lambda k: random.randint(k, (), 0, 7))(action_keys)

        # Step all environments
        new_keys, new_states, _, rewards, dones, _ = batched_step(
            TETROMINOES,
            keys,
            states,
            actions,
            config=CONFIG
        )

        # Increment total steps by number of active (non-done) environments
        new_total_steps = total_steps + BATCH_SIZE

        # Reset done environments
        rng, reset_key = random.split(rng)
        reset_keys = random.split(reset_key, BATCH_SIZE)
        reset_states = batched_reset(
            TETROMINOES,
            reset_keys,
            config=CONFIG,
            batch_size=BATCH_SIZE
        )[1]

        final_states = jax.vmap(lambda state, done, reset_state: jax.lax.cond(
            done,
            lambda _: reset_state,
            lambda _: state,
            operand=None
        ))(new_states, new_states.game_over, reset_states)

        return new_keys, final_states, new_total_steps, rng

    def cond_fun(carry):
        keys, states, total_steps, rng = carry
        return total_steps < num_steps

    final_keys, final_states, total_steps, final_rng = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (keys, states, total_steps, key)
    )

    # Calculate average steps per environment
    avg_steps = total_steps / BATCH_SIZE

    return final_rng, total_steps, avg_steps

@jax.jit
def run_batched_jax_environment(key: chex.PRNGKey, num_steps: int) -> Tuple[chex.PRNGKey, int, float]:
    return play_parallel_episodes_jax(key, num_steps)

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
    num_steps = 50000 * BATCH_SIZE  # Total steps across all environments
    print(f"Running benchmark for {num_steps} total steps ({num_steps / BATCH_SIZE:.0f} steps per environment)")

    print("\nRunning batched JAX environment...")
    key = random.PRNGKey(0)

    # Warm-up run
    key, steps_taken, avg_steps = run_batched_jax_environment(key, 10 * BATCH_SIZE)
    jax.block_until_ready(key)

    # Actual benchmark
    start_time = time.time()
    key, total_steps, avg_steps = run_batched_jax_environment(key, num_steps)
    jax.block_until_ready(key)
    end_time = time.time()

    total_time = end_time - start_time
    steps_per_second = total_steps / total_time
    time_per_step = total_time / total_steps

    print(f"Total time: {total_time:.3f} seconds")
    print(f"Total steps: {total_steps:,}")
    print(f"Steps per second: {steps_per_second:,.0f}")
    print(f"Time per step: {time_per_step:.10f} seconds")
    print(f"Average steps per environment: {avg_steps:.1f}")

if __name__ == "__main__":
    main()


# 0.0000088     (original)
# 0.0000189021  bs 1
# 0.0000033981  bs 32
# 0.0000035687  bs 128
# 0.0000192934  bs 1    (accurate steps counter)
# 0.0000000285  bs 128  (accurate steps counter)
# 0.0000000093  bs 512  (accurate steps counter)
# 0.0000111649  original (same logic)

# 0.0000033763  bs 128  obs
# 0.0000033565  bs 128  no obs