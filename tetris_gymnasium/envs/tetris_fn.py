# step.py

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax import random

from tetris_gymnasium.functional.core import (
    EnvConfig,
    State,
    check_game_over,
    collision,
    create_board,
    get_initial_x_y,
    graviy_step,
    hard_drop,
    lock_active_tetromino,
)
from tetris_gymnasium.functional.queue import (
    CreateQueueFunction,
    QueueFunction,
    bag_queue_get_next_element,
    create_bag_queue,
)
from tetris_gymnasium.functional.tetrominoes import Tetrominoes, get_tetromino_matrix


def step(
    const: Tetrominoes,
    key: chex.PRNGKey,
    state: State,
    action: int,
    config: EnvConfig,
    queue_fn: QueueFunction = bag_queue_get_next_element,
) -> Tuple[chex.PRNGKey, State, float, bool, dict]:
    x, y, rotation = state.x, state.y, state.rotation
    board = state.board
    active_tetromino_matrix = get_tetromino_matrix(
        const, state.active_tetromino, rotation
    )

    def move_left():
        return jax.lax.cond(
            ~collision(board, active_tetromino_matrix, x - 1, y),
            lambda: x - 1,
            lambda: x,
        )

    def move_right():
        return jax.lax.cond(
            ~collision(board, active_tetromino_matrix, x + 1, y),
            lambda: x + 1,
            lambda: x,
        )

    def move_down():
        return jax.lax.cond(
            ~collision(board, active_tetromino_matrix, x, y + 1),
            lambda: y + 1,
            lambda: y,
        )

    def rotate_clockwise():
        new_rotation = (rotation + 1) % 4
        new_matrix = get_tetromino_matrix(const, state.active_tetromino, new_rotation)
        return jax.lax.cond(
            ~collision(board, new_matrix, x, y),
            lambda: (new_rotation, new_matrix),
            lambda: (rotation, active_tetromino_matrix),
        )

    def rotate_counterclockwise():
        new_rotation = (rotation - 1) % 4
        new_matrix = get_tetromino_matrix(const, state.active_tetromino, new_rotation)
        return jax.lax.cond(
            ~collision(board, new_matrix, x, y),
            lambda: (new_rotation, new_matrix),
            lambda: (rotation, active_tetromino_matrix),
        )

    x = jax.lax.switch(
        action,
        [move_left, move_right, lambda: x, lambda: x, lambda: x, lambda: x, lambda: x],
    )
    y = jax.lax.switch(
        action,
        [
            lambda: y,
            lambda: y,
            move_down,
            lambda: y,
            lambda: y,
            lambda: y,
            lambda: hard_drop(board, active_tetromino_matrix, x, y),
        ],
    )
    rotation, active_tetromino_matrix = jax.lax.switch(
        action,
        [
            lambda: (rotation, active_tetromino_matrix),
            lambda: (rotation, active_tetromino_matrix),
            lambda: (rotation, active_tetromino_matrix),
            rotate_clockwise,
            rotate_counterclockwise,
            lambda: (rotation, active_tetromino_matrix),
            lambda: (rotation, active_tetromino_matrix),
        ],
    )

    # Check if the tetromino should be locked
    y_gravity = graviy_step(const, board, state.active_tetromino, rotation, x, y)
    should_lock = y_gravity == y

    state = State(
        board=board,
        active_tetromino=state.active_tetromino,
        rotation=rotation,
        x=x,
        y=y_gravity,
        queue=state.queue,
        queue_index=state.queue_index,
        game_over=False,
        score=state.score,
    )

    # If should lock or it's a hard drop, commit the tetromino
    new_state, new_key = jax.lax.cond(
        should_lock | (action == 6),
        lambda: place_active_tetromino(config, const, state, queue_fn, key),
        lambda: (state, key),
    )

    return (
        key,
        new_state,
        new_state.score - state.score,
        new_state.game_over,
        {}, # info
    )


def reset(
    tetromiones: Tetrominoes,
    key: chex.PRNGKey,
    config: EnvConfig,
    create_queue_fn: CreateQueueFunction = create_bag_queue,
    queue_fn: QueueFunction = bag_queue_get_next_element,
) -> Tuple[chex.PRNGKey, State]:
    board = create_board(config, tetromiones)

    key, subkey = random.split(key)
    queue, queue_index = create_queue_fn(config, key)
    active_tetromino, queue, queue_index, key = queue_fn(
        config, queue, queue_index, key
    )

    x, y = get_initial_x_y(config, tetromiones, active_tetromino)

    state = State(
        board=board,
        active_tetromino=active_tetromino,
        rotation=0,
        x=x,
        y=y,
        queue=queue,
        queue_index=queue_index,
        game_over=False,
        score=jnp.uint8(0),
    )

    return key, state


def place_active_tetromino(
    config: EnvConfig,
    tetrominoes: Tetrominoes,
    state: State,
    queue_fn: QueueFunction,
    key: chex.PRNGKey,
) -> Tuple[State, chex.PRNGKey]:
    # Commit the active tetromino
    new_board, reward = lock_active_tetromino(
        config,
        tetrominoes,
        state.board,
        state.active_tetromino,
        state.rotation,
        state.x,
        state.y,
    )

    # Spawn a new tetromino
    new_active_tetromino, new_queue, new_queue_index, new_key = queue_fn(
        config, state.queue, state.queue_index, key
    )
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
        game_over=game_over,
        score=state.score + reward,
    )

    return new_state, new_key
