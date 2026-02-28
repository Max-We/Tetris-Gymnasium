"""Minimal example: random agent using the JAX functional Tetris environment."""

import jax
import jax.numpy as jnp

from tetris_gymnasium.envs.tetris_fn import reset, step
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

config = EnvConfig(width=10, height=20, padding=4, queue_size=7)

key = jax.random.PRNGKey(42)
key, state, observation = reset(TETROMINOES, key, config)

terminated = False
total_reward = 0.0

while not terminated:
    key, subkey = jax.random.split(key)
    action = int(jax.random.randint(subkey, shape=(), minval=0, maxval=7))
    state, observation, reward, terminated, info = step(
        TETROMINOES, state, action, config
    )
    total_reward += float(reward)

print(f"Game over! Score: {total_reward:.0f}")
