"""Wrappers for the Tetris environment implemented as pure functions."""
from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax import random, vmap

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
    project_tetromino,
)
from tetris_gymnasium.functional.queue import (
    CreateQueueFunction,
    QueueFunction,
    bag_queue_get_next_element,
    create_bag_queue,
)
from tetris_gymnasium.functional.tetrominoes import Tetrominoes, get_tetromino_matrix


def get_observation(
    board, x, y, active_tetromino, rotation, tetrominoes: Tetrominoes, config: EnvConfig
) -> chex.Array:
    tetromino_matrix = get_tetromino_matrix(tetrominoes, active_tetromino, rotation)

    # convert board to values 0 1 (0 if 0, 1 otherwise)
    board = jnp.where(board > 0, 1, 0).astype(jnp.int8)

    board = project_tetromino(
        board, tetromino_matrix, x, y, -1  # display falling tetromino
    )
    return board[0 : -config.padding, config.padding : -config.padding]


def step(
    tetrominoes: Tetrominoes,
    key: chex.PRNGKey,
    state: State,
    action: int,
    config: EnvConfig,
    queue_fn: QueueFunction = bag_queue_get_next_element,
) -> Tuple[chex.PRNGKey, State, chex.Array, float, bool, dict]:
    """Performs a single step in the Tetris environment.

    Args:
        tetrominoes: Tetrominoes object containing tetromino configurations.
        key: Random number generator key.
        state: Current state of the environment.
        action: Integer representing the action to take.
        config: Environment configuration.
        queue_fn: Function to get the next element from the queue.

    Returns:
        A tuple containing:
        - Updated random number generator key
        - New state after the action
        - Reward obtained from the action
        - Boolean indicating if the game is over
        - Dictionary containing additional information
    """
    x, y, rotation = state.x, state.y, state.rotation
    board = state.board
    active_tetromino_matrix = get_tetromino_matrix(
        tetrominoes, state.active_tetromino, rotation
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
        new_matrix = get_tetromino_matrix(
            tetrominoes, state.active_tetromino, new_rotation
        )
        return jax.lax.cond(
            ~collision(board, new_matrix, x, y),
            lambda: (new_rotation, new_matrix),
            lambda: (rotation, active_tetromino_matrix),
        )

    def rotate_counterclockwise():
        new_rotation = (rotation - 1) % 4
        new_matrix = get_tetromino_matrix(
            tetrominoes, state.active_tetromino, new_rotation
        )
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
            rotate_counterclockwise,
            rotate_clockwise,
            lambda: (rotation, active_tetromino_matrix),
            lambda: (rotation, active_tetromino_matrix),
        ],
    )

    # Check if the tetromino should be locked
    y_gravity = jax.lax.cond(
        config.gravity_enabled,
        lambda: graviy_step(tetrominoes, board, state.active_tetromino, rotation, x, y),
        lambda: y,
    )
    should_lock = (y_gravity == y) & config.gravity_enabled

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
        lambda: place_active_tetromino(config, tetrominoes, state, queue_fn, key),
        lambda: (state, key),
    )

    new_observation = get_observation(
        new_state.board,
        new_state.x,
        new_state.y,
        new_state.active_tetromino,
        new_state.rotation,
        tetrominoes,
        config,
    )

    return (
        key,
        new_state,
        new_observation,
        new_state.score - state.score,
        new_state.game_over,
        {},  # info
    )


def reset(
    tetrominoes: Tetrominoes,
    key: chex.PRNGKey,
    config: EnvConfig,
    create_queue_fn: CreateQueueFunction = create_bag_queue,
    queue_fn: QueueFunction = bag_queue_get_next_element,
) -> Tuple[chex.PRNGKey, State, chex.Array]:
    """Resets the Tetris environment to its initial state.

    Args:
        tetrominoes: Tetrominoes object containing tetromino configurations.
        key: Random number generator key.
        config: Environment configuration.
        create_queue_fn: Function to create the initial queue.
        queue_fn: Function to get the next element from the queue.

    Returns:
        A tuple containing:
        - Updated random number generator key
        - Initial state of the environment
    """
    board = create_board(config, tetrominoes)

    key, subkey = random.split(key)
    queue, queue_index = create_queue_fn(config, key)
    active_tetromino, queue, queue_index, key = queue_fn(
        config, queue, queue_index, key
    )

    x, y = get_initial_x_y(config, tetrominoes, active_tetromino)

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

    observation = get_observation(
        state.board,
        state.x,
        state.y,
        state.active_tetromino,
        state.rotation,
        tetrominoes,
        config,
    )

    return key, state, observation


def place_active_tetromino(
    config: EnvConfig,
    tetrominoes: Tetrominoes,
    state: State,
    queue_fn: QueueFunction,
    key: chex.PRNGKey,
) -> Tuple[State, chex.PRNGKey]:
    """Places the active tetromino on the board and updates the game state.

    Args:
        config: Environment configuration.
        tetrominoes: Tetrominoes object containing tetromino configurations.
        state: Current state of the environment.
        queue_fn: Function to get the next element from the queue.
        key: Random number generator key.

    Returns:
        A tuple containing:
        - Updated state after placing the tetromino
        - Updated random number generator key
    """
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


def batched_step(
    tetrominoes: Tetrominoes,
    keys: chex.PRNGKey,  # [B, 2]
    states: State,
    actions: chex.Array,  # [B]
    *,  # Force config to be a keyword argument
    config: EnvConfig,
    queue_fn: QueueFunction = bag_queue_get_next_element,
) -> Tuple[chex.PRNGKey, State, chex.Array, chex.Array, chex.Array, dict]:
    """Vectorized version of step function that handles batches of states."""

    # Create a partial function with static config
    step_partial = partial(step, tetrominoes, config=config, queue_fn=queue_fn)

    # Combine vmap and jit with static config
    batched_step_fn = jax.jit(
        vmap(
            step_partial,
            in_axes=(0, 0, 0),  # Batch key, state, and action
            out_axes=(0, 0, 0, 0, 0, None),  # Batch all outputs except info dict
        ),
        static_argnames=["config"],
    )

    return batched_step_fn(keys, states, actions)


def batched_reset(
    tetrominoes: Tetrominoes,
    keys: chex.PRNGKey,  # [B, 2]
    *,  # Force config to be a keyword argument
    config: EnvConfig,
    create_queue_fn: CreateQueueFunction = create_bag_queue,
    queue_fn: QueueFunction = bag_queue_get_next_element,
    batch_size: int = 1,
) -> Tuple[chex.PRNGKey, chex.Array, State]:
    """Vectorized version of reset function that handles batches."""

    # Create a partial function with static config
    reset_partial = partial(
        reset,
        tetrominoes,
        config=config,
        create_queue_fn=create_queue_fn,
        queue_fn=queue_fn,
    )

    # Combine vmap and jit with static config
    batched_reset_fn = jax.jit(
        vmap(
            reset_partial,
            in_axes=(0,),  # Batch only the key
            out_axes=(0, 0, 0),  # Batch both outputs
        ),
        static_argnames=["config"],
    )

    return batched_reset_fn(keys)


ACTION_ID_TO_NAME = {
    0: "move_left",
    1: "move_right",
    2: "move_down",
    3: "rotate_clockwise",
    4: "rotate_counterclockwise",
    5: "do_nothing",
    6: "hard_drop",
}
