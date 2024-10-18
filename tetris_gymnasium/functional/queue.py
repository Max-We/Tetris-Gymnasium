
from typing import Tuple, Callable

import chex
import jax
import jax.numpy as jnp

from tetris_gymnasium.functional.core import EnvConfig

# This is the type signature for a function that gets the next element from a queue
QueueFunction = Callable[[EnvConfig, chex.Array, int, chex.PRNGKey], Tuple[int, chex.Array, int, chex.PRNGKey]]

# Bag functions
def create_bag_queue(config: EnvConfig, key: chex.Array) -> Tuple[chex.Array, int]:
    return jax.random.permutation(key, jnp.arange(config.queue_size)), 0

def bag_queue_get_next_element(config: EnvConfig, queue: chex.Array, queue_index: int, key: chex.PRNGKey) -> Tuple[
    int, chex.Array, int, chex.PRNGKey]:

    def refill():
        new_key, subkey = jax.random.split(key)
        new_queue, new_index = create_bag_queue(config, subkey)
        return new_queue[new_index], new_queue, new_index + 1, new_key

    def get_current():
        return queue[queue_index], queue, queue_index + 1, key

    return jax.lax.cond(
        queue_index >= config.queue_size,
        refill,
        get_current
    )

# Uniform random queue functions
def create_uniform_queue(config: EnvConfig, key: chex.PRNGKey) -> Tuple[chex.Array, int]:
    return jax.random.randint(key, (config.queue_size,), 0, config.queue_size-1), 0

def uniform_queue_get_next_element(config: EnvConfig, queue: chex.Array, queue_index: int, key: chex.PRNGKey) -> Tuple[
    int, chex.Array, int, chex.PRNGKey]:

    def refill():
        new_key, subkey = jax.random.split(key)
        new_queue, new_index = create_uniform_queue(config, subkey)
        return new_queue[new_index], new_queue, new_index + 1, new_key

    def get_current():
        return queue[queue_index], queue, queue_index + 1, key

    return jax.lax.cond(
        queue_index >= config.queue_size,
        refill,
        get_current
    )
