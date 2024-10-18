# step.py

import chex
import jax
import jax.numpy as jnp
from jax import random


from typing import Tuple, Optional, Callable

from tetris_gymnasium.functional.game_logic import EnvConfig, State, create_board, collision, project_tetromino, \
    clear_filled_rows, get_initial_x_y, hard_drop, commit_active_tetromino
from tetris_gymnasium.functional.tetrominoes import get_tetromino_matrix, Tetrominoes


def step(
    const: Tetrominoes,
    key: chex.PRNGKey,
    state: State,
    action: int,
    config: EnvConfig,
    queue_function: Optional[Callable] = None,
    holder_function: Optional[Callable] = None
) -> Tuple[chex.PRNGKey, State, float, bool, dict]:
    x, y, rotation = state.x, state.y, state.rotation
    board = state.board
    active_tetromino_matrix = get_tetromino_matrix(const, state.active_tetromino, rotation)
    reward = 0.0
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
        if holder_function is not None:
            new_active, new_holder, new_has_swapped = holder_function(state.active_tetromino, state.holder, state.has_swapped)
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

    # If should lock or it's a hard drop, commit the tetromino
    key, new_state, lock_reward, game_over = jax.lax.cond(
        should_lock | (action == 6),
        lambda: commit_active_tetromino(config, const, State(
            board=board,
            active_tetromino=state.active_tetromino,
            rotation=rotation,
            x=x,
            y=y,
            queue=state.queue,
            holder=state.holder,
            has_swapped=state.has_swapped,
            game_over=False,
            score=state.score
        ), key),
        lambda: (key, State(
            board=board,
            active_tetromino=state.active_tetromino,
            rotation=rotation,
            x=x,
            y=y,
            queue=state.queue,
            holder=state.holder,
            has_swapped=state.has_swapped,
            game_over=False,
            score=state.score
        ), jnp.float32(0.0), False)
    )
    #     return key, new_state, reward, game_over

    reward += lock_reward

    return key, new_state, reward, game_over, {"lines_cleared": lines_cleared}

def reset(
    const: EnvConfig,
    key: chex.PRNGKey,
    config: EnvConfig
) -> Tuple[chex.PRNGKey, State]:
    board = create_board(config, const)
    key, subkey = random.split(key)

    if config.queue_size > 0:
        queue = random.randint(subkey, (config.queue_size,), 0, len(const.tetromino_ids))
        active_tetromino = queue[0]
    else:
        queue = None
        active_tetromino = random.randint(subkey, (), 0, len(const.tetromino_ids))

    x, y = get_initial_x_y(config, const, active_tetromino)

    state = State(
        board=board,
        active_tetromino=active_tetromino,
        rotation=0,
        x=x,
        y=y,
        queue=queue,
        holder=-1,
        has_swapped=False,
        game_over=False,
        score=0
    )

    return key, state