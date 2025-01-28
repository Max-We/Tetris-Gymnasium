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
    board,
    x,
    y,
    active_tetromino,
    rotation,
    game_over,
    tetrominoes: Tetrominoes,
    config: EnvConfig,
) -> chex.Array:
    """Returns the observation of the environment."""
    tetromino_matrix = get_tetromino_matrix(tetrominoes, active_tetromino, rotation)
    board = jnp.where(board > 0, 1, 0).astype(jnp.int8)

    result = jax.lax.cond(
        ~game_over,
        lambda _: project_tetromino(board, tetromino_matrix, x, y, -1),
        lambda _: board,
        None,
    )

    return result[0 : -config.padding, config.padding : -config.padding]


def update_state(action, config, state, queue_fn, tetrominoes):
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
        new_y = jax.lax.cond(
            ~collision(board, active_tetromino_matrix, x, y + 1),
            lambda: y + 1,
            lambda: y,
        )
        move_reward = jnp.int32(new_y - y)
        return new_y, move_reward

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
    y, drop_reward = jax.lax.switch(
        action,
        [
            lambda: (y, 0),
            lambda: (y, 0),
            move_down,
            lambda: (y, 0),
            lambda: (y, 0),
            lambda: (y, 0),
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
    y_gravity = jax.lax.cond(
        config.gravity_enabled,
        lambda: graviy_step(tetrominoes, board, state.active_tetromino, rotation, x, y),
        lambda: y,
    )
    should_lock = (y_gravity == y) & config.gravity_enabled

    # Create intermediate state with updated position and rotation
    intermediate_state = State(
        rng_key=state.rng_key,
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
    # Handle locking and new piece spawning
    new_state, lock_reward, lines_cleared = jax.lax.cond(
        (should_lock | (action == 6)),
        lambda: place_active_tetromino(
            config, tetrominoes, intermediate_state, queue_fn
        ),
        lambda: (intermediate_state, 0, 0),
    )
    # Update score
    new_state = new_state.replace(score=new_state.score + drop_reward + lock_reward)

    return new_state, lines_cleared


def step(
    tetrominoes: Tetrominoes,
    state: State,
    action: int,
    config: EnvConfig,
    queue_fn: QueueFunction = bag_queue_get_next_element,
) -> Tuple[State, chex.Array, float, bool, dict]:
    """Performs a single step in the Tetris environment."""

    new_state, lines_cleared = jax.lax.cond(
        state.game_over,
        lambda _: (state, 0),
        lambda _: update_state(action, config, state, queue_fn, tetrominoes),
        None,
    )

    new_observation = get_observation(
        new_state.board,
        new_state.x,
        new_state.y,
        new_state.active_tetromino,
        new_state.rotation,
        new_state.game_over,
        tetrominoes,
        config,
    )

    return (
        new_state,  # state
        new_observation,  # observation
        new_state.score - state.score,  # reward
        new_state.game_over,  # terminated
        {"lines_cleared": lines_cleared},  # info
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
        rng_key=subkey,
        board=board,
        active_tetromino=active_tetromino,
        rotation=0,
        x=x,
        y=y,
        queue=queue,
        queue_index=queue_index,
        game_over=False,
        score=jnp.float32(0),
    )

    observation = get_observation(
        state.board,
        state.x,
        state.y,
        state.active_tetromino,
        state.rotation,
        state.game_over,
        tetrominoes,
        config,
    )

    return key, state, observation


def place_active_tetromino(
    config: EnvConfig,
    tetrominoes: Tetrominoes,
    state: State,
    queue_fn: QueueFunction,
) -> Tuple[State, chex.Array, chex.Array]:
    """Places the active tetromino on the board and updates the game state."""
    # Commit the active tetromino
    new_board, reward, lines_cleared = lock_active_tetromino(
        config,
        tetrominoes,
        state.board,
        state.active_tetromino,
        state.rotation,
        state.x,
        state.y,
    )

    # Spawn a new tetromino
    new_active_tetromino, new_queue, new_queue_index, _ = queue_fn(
        config, state.queue, state.queue_index, state.rng_key
    )
    new_x, new_y = get_initial_x_y(config, tetrominoes, new_active_tetromino)
    new_rotation = 0

    # Check if the game is over
    game_over = check_game_over(
        tetrominoes, new_board, new_active_tetromino, new_rotation, new_x, new_y
    )

    new_rng_key = random.split(state.rng_key)[0]
    new_state = State(
        board=new_board,
        active_tetromino=new_active_tetromino,
        rotation=new_rotation,
        x=new_x,
        y=new_y,
        queue=new_queue,
        queue_index=new_queue_index,
        game_over=game_over,
        score=state.score,
        rng_key=new_rng_key,
    )

    return new_state, reward, lines_cleared


def batched_step(
    tetrominoes: Tetrominoes,
    states: State,
    actions: chex.Array,  # [B]
    *,  # Force config to be a keyword argument
    config: EnvConfig,
    queue_fn: QueueFunction = bag_queue_get_next_element,
) -> Tuple[State, chex.Array, chex.Array, chex.Array, dict]:
    """Vectorized version of step function that handles batches of states."""
    step_partial = partial(step, tetrominoes, config=config, queue_fn=queue_fn)
    batched_step_fn = jax.jit(
        vmap(
            step_partial,
            in_axes=(0, 0),  # Batch state and action
            out_axes=(0, 0, 0, 0, 0),
        ),
        static_argnames=["config"],
    )

    return batched_step_fn(states, actions)


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
