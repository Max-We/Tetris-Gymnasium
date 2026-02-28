"""Tests for queue creation and retrieval."""

import jax
import jax.numpy as jnp

from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.queue import (
    bag_queue_get_next_element,
    create_bag_queue,
    create_uniform_queue,
    uniform_queue_get_next_element,
)

CONFIG = EnvConfig(
    width=10,
    height=20,
    padding=4,
    queue_size=7,
    gravity_enabled=True,
)


class TestBagQueue:
    def test_create_bag_queue_length(self):
        key = jax.random.PRNGKey(0)
        queue, index = create_bag_queue(CONFIG, key)
        assert queue.shape == (7,)
        assert int(index) == 0

    def test_create_bag_queue_contains_all_indices(self):
        key = jax.random.PRNGKey(0)
        queue, _ = create_bag_queue(CONFIG, key)
        assert set(queue.tolist()) == set(range(7))

    def test_deterministic_with_same_key(self):
        key = jax.random.PRNGKey(42)
        q1, _ = create_bag_queue(CONFIG, key)
        q2, _ = create_bag_queue(CONFIG, key)
        assert jnp.array_equal(q1, q2)

    def test_get_next_element_increments_index(self):
        key = jax.random.PRNGKey(0)
        queue, index = create_bag_queue(CONFIG, key)
        elem, new_queue, new_index, _ = bag_queue_get_next_element(
            CONFIG, queue, index, key
        )
        assert int(new_index) == int(index) + 1
        assert int(elem) == int(queue[index])

    def test_full_bag_consumption_all_seven(self):
        key = jax.random.PRNGKey(0)
        queue, index = create_bag_queue(CONFIG, key)
        elements = []
        for _ in range(7):
            elem, queue, index, key = bag_queue_get_next_element(
                CONFIG, queue, index, key
            )
            elements.append(int(elem))
        assert set(elements) == set(range(7))

    def test_refill_after_exhaustion(self):
        key = jax.random.PRNGKey(0)
        queue, index = create_bag_queue(CONFIG, key)
        for _ in range(7):
            _, queue, index, key = bag_queue_get_next_element(CONFIG, queue, index, key)
        # Next call should refill
        elem, new_queue, new_index, _ = bag_queue_get_next_element(
            CONFIG, queue, index, key
        )
        assert 0 <= int(elem) < 7
        assert int(new_index) == 1  # just consumed first of new bag


class TestUniformQueue:
    def test_create_uniform_queue_values_in_range(self):
        key = jax.random.PRNGKey(0)
        queue, index = create_uniform_queue(CONFIG, key)
        assert queue.shape == (7,)
        assert int(index) == 0
        # uniform uses [0, queue_size-1)
        assert jnp.all(queue >= 0) and jnp.all(queue < CONFIG.queue_size - 1)

    def test_uniform_get_next_element(self):
        key = jax.random.PRNGKey(0)
        queue, index = create_uniform_queue(CONFIG, key)
        elem, _, new_index, _ = uniform_queue_get_next_element(
            CONFIG, queue, index, key
        )
        assert int(new_index) == 1
        assert int(elem) == int(queue[0])
