# step.py

from typing import Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jax import random

from tetris_gymnasium.functional.core import (
    EnvConfig,
    State,
    collision,
    lock_active_tetromino,
    create_board,
    get_initial_x_y,
    hard_drop,
    check_game_over,
)
from tetris_gymnasium.functional.queue import QueueFunction, create_bag_queue, bag_queue_get_next_element, \
    uniform_queue_get_next_element, CreateQueueFunction
from tetris_gymnasium.functional.tetrominoes import Tetrominoes, get_tetromino_matrix


def step(
    const: Tetrominoes,
    key: chex.PRNGKey,
    state: State,
    action: int,
    config: EnvConfig,
    queue_fn: QueueFunction = bag_queue_get_next_element,
    holder_fn: Optional[Callable] = None
) -> Tuple[chex.PRNGKey, State, float, bool, dict]:
    x, y, rotation = state.x, state.y, state.rotation
    board = state.board
    active_tetromino_matrix = get_tetromino_matrix(const, state.active_tetromino, rotation)
    lines_cleared = 0

    def move_left():
        return jax.lax.cond(
            ~collision(board, active_tetromino_matrix, x - 1, y),
            lambda: x - 1,
            lambda: x
        )

    def move_right():
        return jax.lax.cond(
            ~collision(board, active_tetromino_matrix, x + 1, y),
            lambda: x + 1,
            lambda: x
        )

    def move_down():
        return jax.lax.cond(
            ~collision(board, active_tetromino_matrix, x, y + 1),
            lambda: y + 1,
            lambda: y
        )

    def rotate_clockwise():
        new_rotation = (rotation + 1) % 4
        new_matrix = get_tetromino_matrix(const, state.active_tetromino, new_rotation)
        return jax.lax.cond(
            ~collision(board, new_matrix, x, y),
            lambda: (new_rotation, new_matrix),
            lambda: (rotation, active_tetromino_matrix)
        )

    def rotate_counterclockwise():
        new_rotation = (rotation - 1) % 4
        new_matrix = get_tetromino_matrix(const, state.active_tetromino, new_rotation)
        return jax.lax.cond(
            ~collision(board, new_matrix, x, y),
            lambda: (new_rotation, new_matrix),
            lambda: (rotation, active_tetromino_matrix)
        )

    def handle_swap():
        if holder_fn is not None:
            new_active, new_holder, new_has_swapped = holder_fn(state.active_tetromino, state.holder, state.has_swapped)
            new_x, new_y = get_initial_x_y(config, const, new_active)
            return new_active, new_holder, new_x, new_y, 0, new_has_swapped
        return state.active_tetromino, state.holder, x, y, rotation, state.has_swapped

    x = jax.lax.switch(action, [move_left, move_right, lambda: x, lambda: x, lambda: x, lambda: x, lambda: x])
    y = jax.lax.switch(action, [lambda: y, lambda: y, move_down, lambda: y, lambda: y, lambda: y, lambda: hard_drop(
        board, active_tetromino_matrix, x, y)])
    rotation, active_tetromino_matrix = jax.lax.switch(action, [
        lambda: (rotation, active_tetromino_matrix),
        lambda: (rotation, active_tetromino_matrix),
        lambda: (rotation, active_tetromino_matrix),
        rotate_clockwise,
        rotate_counterclockwise,
        lambda: (rotation, active_tetromino_matrix),
        lambda: (rotation, active_tetromino_matrix)
    ])

    state.active_tetromino, state.holder, x, y, rotation, state.has_swapped = jax.lax.cond(
        action == 5,
        handle_swap,
        lambda: (state.active_tetromino, state.holder, x, y, rotation, state.has_swapped)
    )

    # Check if the tetromino should be locked
    should_lock = collision(board, active_tetromino_matrix, x, y + 1)

    state = State(
        board = board,
        active_tetromino = state.active_tetromino,
        rotation = rotation,
        x = x,
        y = y,
        queue = state.queue,
        queue_index = state.queue_index,
        holder = state.holder,
        has_swapped = state.has_swapped,
        game_over = False,
        score = state.score
    )

    # If should lock or it's a hard drop, commit the tetromino
    new_state, new_key = jax.lax.cond(
        should_lock | (action == 6),
        lambda: place_active_tetromino(config, const, state, queue_fn, key),
        lambda: (state, key)
    )

    return key, new_state, new_state.score - state.score, new_state.game_over, {"lines_cleared": lines_cleared}

def reset(
    const: EnvConfig,
    key: chex.PRNGKey,
    config: EnvConfig,
    create_queue_fn: CreateQueueFunction= create_bag_queue
) -> Tuple[chex.PRNGKey, State]:
    board = create_board(config, const)
    key, subkey = random.split(key)

    if config.queue_size > 0:
        queue = random.randint(subkey, (config.queue_size,), 0, len(const.ids))
        active_tetromino = queue[0]
    else:
        queue = None
        active_tetromino = random.randint(subkey, (), 0, len(const.ids))

    x, y = get_initial_x_y(config, const, active_tetromino)

    queue, queue_index = create_queue_fn(config, key)

    state = State(
        board=board,
        active_tetromino=active_tetromino,
        rotation=0,
        x=x,
        y=y,
        queue=queue,
        queue_index=queue_index,
        holder=-1,
        has_swapped=False,
        game_over=False,
        score=jnp.uint8(0)
    )

    return key, state

def place_active_tetromino(
    config: EnvConfig,
    tetrominoes: Tetrominoes,
    state: State,
    queue_fn: QueueFunction,
    key: chex.PRNGKey
) -> Tuple[State, chex.PRNGKey]:
    # Commit the active tetromino
    new_board, reward = lock_active_tetromino(
        config, tetrominoes, state.board, state.active_tetromino, state.rotation, state.x, state.y
    )

    # Spawn a new tetromino
    new_active_tetromino, new_queue, new_queue_index, new_key = queue_fn(config, state.queue, state.queue_index, key)
    new_x, new_y = get_initial_x_y(config, tetrominoes, new_active_tetromino)
    new_rotation = 0

    # Check if the game is over
    game_over = check_game_over(
        tetrominoes, new_board, new_active_tetromino, new_rotation, new_x, new_y
    )

    new_state = State(
        board=new_board,
        active_tetromino=new_active_tetromino,
        rotation=new_rotation,
        x=new_x,
        y=new_y,
        queue=new_queue,
        queue_index=new_queue_index,
        holder=state.holder,
        has_swapped=state.has_swapped,
        game_over=game_over,
        score=state.score + reward
    )

    return new_state, new_key
