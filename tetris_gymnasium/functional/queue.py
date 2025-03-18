"""Queue functions for Tetris environments."""

from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp

from tetris_gymnasium.functional.core import EnvConfig

# This is the type signature for a function that gets the next element from a queue
QueueFunction = Callable[
    [EnvConfig, chex.Array, int, chex.PRNGKey],
    Tuple[int, chex.Array, int, chex.PRNGKey],
]
CreateQueueFunction = Callable[[EnvConfig, chex.PRNGKey], Tuple[chex.Array, int]]


# Bag functions
def create_bag_queue(config: EnvConfig, key: chex.Array) -> Tuple[chex.Array, int]:
    """Creates a bag queue for Tetris pieces.

    This function generates a random permutation of piece indices to ensure
    a fair distribution of pieces.

    Args:
        config: The environment configuration.
        key: The random number generator key.

    Returns:
        A tuple containing:
        - The queue as a permuted array of piece indices.
        - The initial queue index (0).
    """
    return jax.random.permutation(key, jnp.arange(config.queue_size)), 0


def bag_queue_get_next_element(
    config: EnvConfig, queue: chex.Array, queue_index: int, key: chex.PRNGKey
) -> Tuple[int, chex.Array, int, chex.PRNGKey]:
    """Retrieves the next element from the bag queue.

    If the queue is exhausted, it generates a new permutation.

    Args:
        config: The environment configuration.
        queue: The current queue of piece indices.
        queue_index: The current index in the queue.
        key: The random number generator key.

    Returns:
        A tuple containing:
        - The next piece index.
        - The updated queue.
        - The updated queue index.
        - The updated random number generator key.
    """

    def refill():
        new_key, subkey = jax.random.split(key)
        new_queue, new_index = create_bag_queue(config, subkey)
        return new_queue[new_index], new_queue, new_index + 1, new_key

    def get_current():
        return queue[queue_index], queue, queue_index + 1, key

    return jax.lax.cond(queue_index >= config.queue_size, refill, get_current)


# Uniform random queue functions
def create_uniform_queue(
    config: EnvConfig, key: chex.PRNGKey
) -> Tuple[chex.Array, int]:
    """Creates a uniform random queue for Tetris pieces.

    This function generates a queue of random piece indices.

    Args:
        config: The environment configuration.
        key: The random number generator key.

    Returns:
        A tuple containing:
        - The queue as an array of random piece indices.
        - The initial queue index (0).
    """
    return jax.random.randint(key, (config.queue_size,), 0, config.queue_size - 1), 0


def uniform_queue_get_next_element(
    config: EnvConfig, queue: chex.Array, queue_index: int, key: chex.PRNGKey
) -> Tuple[int, chex.Array, int, chex.PRNGKey]:
    """Retrieves the next element from the uniform random queue.

    If the queue is exhausted, it generates a new random queue.

    Args:
        config: The environment configuration.
        queue: The current queue of piece indices.
        queue_index: The current index in the queue.
        key: The random number generator key.

    Returns:
        A tuple containing:
        - The next piece index.
        - The updated queue.
        - The updated queue index.
        - The updated random number generator key.
    """

    def refill():
        new_key, subkey = jax.random.split(key)
        new_queue, new_index = create_uniform_queue(config, subkey)
        return new_queue[new_index], new_queue, new_index + 1, new_key

    def get_current():
        return queue[queue_index], queue, queue_index + 1, key

    return jax.lax.cond(queue_index >= config.queue_size, refill, get_current)
