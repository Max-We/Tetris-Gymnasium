# game_logic.py

from typing import NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from tetris_gymnasium.functional.tetrominoes import Tetrominoes, get_tetromino_matrix


# Configuration and State definitions
class EnvConfig(NamedTuple):
    width: int
    height: int
    padding: int
    queue_size: int


@chex.dataclass
class State:
    board: chex.Array
    active_tetromino: int
    rotation: int
    x: int
    y: int
    queue: chex.Array  # Fixed-size array for the queue
    queue_index: int
    # holder: Optional[int]
    game_over: bool
    score: int


# Utility functions
def create_board(config: EnvConfig, tetrominoes: Tetrominoes) -> chex.Array:
    empty_board = jnp.zeros((config.height, config.width), dtype=jnp.uint8)
    padded_board = jnp.pad(
        empty_board,
        ((0, config.padding), (config.padding, config.padding)),
        mode="constant",
        constant_values=tetrominoes.base_pixels[1],
    )
    return padded_board


def get_initial_x_y(
    config: EnvConfig, tetrominoes: Tetrominoes, active_tetromino: int
) -> Tuple[int, int]:
    x = (config.width + config.padding * 2) // 2 - get_tetromino_matrix(
        tetrominoes, active_tetromino, 0
    ).shape[1] // 2
    y = 0
    return x, y


def collision(board: chex.Array, tetromino: chex.Array, x: int, y: int) -> chex.Array:
    h, w = tetromino.shape
    board_section = jax.lax.dynamic_slice(board, (y, x), (h, w))
    return jnp.any((board_section > 0) & (tetromino > 0))


def project_tetromino(
    board: chex.Array, tetromino: chex.Array, x: int, y: int, tetromino_id: int
) -> chex.Array:
    update = jax.lax.dynamic_update_slice(
        jnp.zeros_like(board), tetromino * tetromino_id, (y, x)
    )
    return board + update


def score(config: EnvConfig, rows_cleared: int) -> jnp.uint8:
    return jnp.uint8((rows_cleared**2) * config.width)


# Core game logic functions
def graviy_step(
    tetrominoes: Tetrominoes,
    board: chex.Array,
    active_tetromino: int,
    rotation: int,
    x: int,
    y: int,
) -> Tuple[chex.Array, int]:
    new_y = jax.lax.cond(
        ~collision(
            board,
            get_tetromino_matrix(tetrominoes, active_tetromino, rotation),
            x,
            y + 1,
        ),
        lambda: y + 1,
        lambda: y,
    )

    return new_y


def clear_filled_rows(
    config: EnvConfig, tetrominoes: Tetrominoes, board: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    filled_rows = jnp.all(board[:, config.padding : -config.padding] > 0, axis=1)
    n_filled = jnp.sum(filled_rows)

    def clear_rows(board):
        height = board.shape[0]

        def body_fn(i, state):
            new_board, write_idx = state
            row = board[i]
            is_filled = filled_rows[i]

            new_board = new_board.at[write_idx].set(
                jnp.where(is_filled, new_board[write_idx], row)
            )
            write_idx = write_idx + (1 - is_filled)

            return new_board, write_idx

        init_board = create_board(config, tetrominoes)
        final_board, _ = jax.lax.fori_loop(0, height, body_fn, (init_board, 0))
        return final_board

    board = jax.lax.cond(n_filled > 0, clear_rows, lambda x: x, board)

    return board, n_filled


def hard_drop(board: chex.Array, tetromino: chex.Array, x: int, y: int) -> int:
    def cond_fun(y):
        return ~collision(board, tetromino, x, y + 1)

    def body_fun(y):
        return y + 1

    return jax.lax.while_loop(cond_fun, body_fun, y)


def lock_active_tetromino(
    config: EnvConfig, tetrominoes: Tetrominoes, board, active_tetromino, rotation, x, y
) -> Tuple[chex.Array, chex.Array]:
    tetromino_matrix = get_tetromino_matrix(tetrominoes, active_tetromino, rotation)
    # place the tetromino on the board
    updated_board = project_tetromino(
        board, tetromino_matrix, x, y, tetrominoes.ids[active_tetromino]
    )
    # clear filled rows
    updated_board, lines_cleared = clear_filled_rows(config, tetrominoes, updated_board)
    # calculate reward
    reward = score(config, lines_cleared)

    return updated_board, reward


def check_game_over(
    tetrominoes: Tetrominoes, board, active_tetromino, rotation, x, y
) -> bool:
    return collision(
        board, get_tetromino_matrix(tetrominoes, active_tetromino, rotation), x, y
    )
