# game_logic.py

import jax
import jax.numpy as jnp
from jax import random
import chex
from typing import Tuple, Optional, Callable, NamedTuple

from tetris_gymnasium.functional.tetrominoes import Tetrominoes, get_tetromino_matrix


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
    queue: Optional[chex.Array]
    holder: Optional[int]
    has_swapped: bool
    game_over: bool
    score: int

def create_board(config: EnvConfig, const: Tetrominoes) -> chex.Array:
    board = jnp.zeros((config.height, config.width), dtype=jnp.uint8)
    return jnp.pad(board, ((0, config.padding), (config.padding, config.padding)), mode='constant', constant_values=const.base_pixels[1])

def collision(board: chex.Array, tetromino: chex.Array, x: int, y: int) -> bool:
    h, w = tetromino.shape
    board_section = jax.lax.dynamic_slice(board, (y, x), (h, w))
    return jnp.any((board_section > 0) & (tetromino > 0))

def project_tetromino(board: chex.Array, tetromino: chex.Array, x: int, y: int, tetromino_id: int) -> chex.Array:
    # h, w = tetromino.shape
    update = jax.lax.dynamic_update_slice(jnp.zeros_like(board), tetromino * tetromino_id, (y, x))
    return board + update

def clear_filled_rows(config: EnvConfig, const: Tetrominoes, board: chex.Array) -> Tuple[chex.Array, int]:
    filled_rows = jnp.all(board[:, config.padding:-config.padding] > 0, axis=1)
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

        init_board = create_board(config, const)
        final_board, _ = jax.lax.fori_loop(0, height, body_fn, (init_board, 0))
        return final_board

    board = jax.lax.cond(n_filled > 0, clear_rows, lambda x: x, board)

    return board, n_filled

def score(config: EnvConfig, rows_cleared: int) -> float:
    return jnp.float32((rows_cleared ** 2) * config.width)

def hard_drop(board: chex.Array, tetromino: chex.Array, x: int, y: int) -> int:
    def cond_fun(y):
        return ~collision(board, tetromino, x, y + 1)

    def body_fun(y):
        return y+1

    final_y = jax.lax.while_loop(cond_fun, body_fun, y)
    return final_y

def update_queue(const: Tetrominoes, queue: chex.Array, key: chex.PRNGKey) -> Tuple[chex.Array, chex.PRNGKey]:
    key, subkey = random.split(key)
    new_tetromino = random.randint(subkey, (), 0, len(const.ids))
    new_queue = jnp.roll(queue, -1)
    new_queue = new_queue.at[-1].set(new_tetromino)
    return new_queue, key

def swap_holder(active_tetromino: int, holder: int, has_swapped: bool) -> Tuple[int, int, bool]:
    return jax.lax.cond(
        ~has_swapped,
        lambda: (holder if holder != -1 else active_tetromino, active_tetromino, True),
        lambda: (active_tetromino, holder, has_swapped)
    )

def commit_active_tetromino(config: EnvConfig, const: Tetrominoes, state: State, key: chex.PRNGKey) -> Tuple[chex.PRNGKey, State, float, bool]:
    tetromino_matrix = get_tetromino_matrix(const, state.active_tetromino, state.rotation)
    board = project_tetromino(state.board, tetromino_matrix, state.x, state.y, const.ids[state.active_tetromino])
    board, lines_cleared = clear_filled_rows(config, const, board)
    reward = score(config, lines_cleared)

    def use_queue(args):
        queue, key = args
        new_queue, new_key = update_queue(const, queue, key)
        return new_queue[0], new_queue, new_key

    def use_random(args):
        _, key = args
        new_key, subkey = random.split(key)
        new_tetromino = random.randint(subkey, (), 0, len(const.ids))
        return new_tetromino, None, new_key

    new_active_tetromino, new_queue, key = use_random((None, key))

    new_x, new_y = get_initial_x_y(config, const, new_active_tetromino)
    new_rotation = 0

    game_over = collision(board, get_tetromino_matrix(const, new_active_tetromino, new_rotation), new_x, new_y)

    new_state = State(
        board=board,
        active_tetromino=new_active_tetromino,
        rotation=new_rotation,
        x=new_x,
        y=new_y,
        queue=state.queue,
        holder=state.holder,
        has_swapped=False,
        game_over=game_over,
        score=state.score + jnp.int32(reward)
    )

    return key, new_state, reward, game_over

def get_initial_x_y(config: EnvConfig, const: Tetrominoes, active_tetromino: int) -> Tuple[int, int]:
    x = (config.width + config.padding * 2) // 2 - get_tetromino_matrix(const, active_tetromino, 0).shape[1] // 2
    y = 0
    return x, y
