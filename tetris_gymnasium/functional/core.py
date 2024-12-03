"""Functional core for Tetris Gymnasium."""
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp

from tetris_gymnasium.functional.tetrominoes import Tetrominoes, get_tetromino_matrix


class EnvConfig(NamedTuple):
    """Configuration for the Tetris environment.

    Attributes:
        width (int): The width of the game board.
        height (int): The height of the game board.
        padding (int): The padding around the game board.
        queue_size (int): The size of the tetromino queue.
    """

    width: int
    height: int
    padding: int
    queue_size: int
    gravity_enabled: bool = True


@chex.dataclass
class State:
    """State of the Tetris environment."""

    board: chex.Array  # [B, H, W]
    active_tetromino: chex.Array  # [B, 1]
    rotation: chex.Array  # [B, 1]
    x: chex.Array  # [B, 1]
    y: chex.Array  # [B, 1]
    queue: chex.Array  # [B, L, 1]
    queue_index: chex.Array  # [B, 1]
    # holder: Optional[int]
    game_over: chex.Array  # [B, 1]
    score: chex.Array  # [B, 1]


# Utility functions
def create_board(config: EnvConfig, tetrominoes: Tetrominoes) -> chex.Array:
    """Creates an empty Tetris board with padding.

    Args:
        config: Environment configuration.
        tetrominoes: Tetrominoes object containing tetromino configurations.

    Returns:
        A 2D array representing the empty Tetris board with padding.
    """
    empty_board = jnp.zeros((config.height, config.width), dtype=jnp.int8)
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
    """Calculates the initial x and y coordinates for a new tetromino.

    Args:
        config: Environment configuration.
        tetrominoes: Tetrominoes object containing tetromino configurations.
        active_tetromino: ID of the active tetromino.

    Returns:
        A tuple containing the initial x and y coordinates.
    """
    x = (config.width + config.padding * 2) // 2 - get_tetromino_matrix(
        tetrominoes, active_tetromino, 0
    ).shape[1] // 2
    y = 0
    return x, y


def collision(board: chex.Array, tetromino: chex.Array, x: int, y: int) -> chex.Array:
    """Checks if there's a collision between the tetromino and the board at the given position.

    Args:
        board: The current state of the Tetris board.
        tetromino: The tetromino to check for collision.
        x: The x-coordinate of the tetromino's position.
        y: The y-coordinate of the tetromino's position.

    Returns:
        A boolean indicating whether there's a collision.
    """
    h, w = tetromino.shape
    board_section = jax.lax.dynamic_slice(board, (y, x), (h, w))
    return jnp.any((board_section > 0) & (tetromino > 0))


def project_tetromino(
    board: chex.Array, tetromino: chex.Array, x: int, y: int, tetromino_id: int
) -> chex.Array:
    """Projects a tetromino onto the board at the given position.

    Args:
        board: The current state of the Tetris board.
        tetromino: The tetromino to project.
        x: The x-coordinate of the tetromino's position.
        y: The y-coordinate of the tetromino's position.
        tetromino_id: The ID of the tetromino.

    Returns:
        The updated board with the projected tetromino.
    """
    update = jax.lax.dynamic_update_slice(
        jnp.zeros_like(board), tetromino * tetromino_id, (y, x)
    )
    return board + update


def score(config: EnvConfig, rows_cleared: int) -> jnp.uint8:
    """Calculates the score based on the number of rows cleared.

    Args:
        config: Environment configuration.
        rows_cleared: The number of rows cleared.

    Returns:
        The calculated score as a uint8.
    """
    standard_clears_reward = jax.lax.cond(
        rows_cleared > 0,
        lambda _: jnp.int32(rows_cleared * 200 - 100),
        lambda _: jnp.int32(0),
        operand=None,
    )
    tetris_reward = jax.lax.cond(
        rows_cleared == 4,
        lambda _: jnp.int32(800),
        lambda _: standard_clears_reward,
        operand=None,
    )
    return tetris_reward


# Core game logic functions
def graviy_step(
    tetrominoes: Tetrominoes,
    board: chex.Array,
    active_tetromino: int,
    rotation: int,
    x: int,
    y: int,
) -> Tuple[chex.Array, int]:
    """Applies gravity to the active tetromino, moving it down if possible.

    Args:
        tetrominoes: Tetrominoes object containing tetromino configurations.
        board: The current state of the Tetris board.
        active_tetromino: ID of the active tetromino.
        rotation: Current rotation of the active tetromino.
        x: Current x-coordinate of the active tetromino.
        y: Current y-coordinate of the active tetromino.

    Returns:
        The new y-coordinate after applying gravity.
    """
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
    """Clears filled rows from the board and returns the updated board and number of cleared rows.

    Args:
        config: Environment configuration.
        tetrominoes: Tetrominoes object containing tetromino configurations.
        board: The current state of the Tetris board.

    Returns:
        A tuple containing the updated board and the number of cleared rows.
    """
    sub_board = board[: -config.padding, config.padding : -config.padding]
    filled_rows_mask = jnp.all(sub_board > 0, axis=1)
    n_filled = jnp.sum(filled_rows_mask)

    def clear_rows(sub_board):
        indices = jnp.arange(config.height)
        uncleared_board_indices = jnp.where(
            filled_rows_mask, -config.height, indices
        )  # -config.height for invalid indices
        cleared_board_indices = jnp.sort(
            uncleared_board_indices
        )  # cleared rows to the top

        # Create new board by referencing the cleared rows
        cleared_sub_board = jnp.take(
            sub_board, cleared_board_indices, axis=0, fill_value=0
        )

        # Add padding to the cleared sub-board
        board = jnp.pad(
            cleared_sub_board,
            ((0, config.padding), (config.padding, config.padding)),
            mode="constant",
            constant_values=tetrominoes.base_pixels[1],
        )
        return board

    board = jax.lax.cond(n_filled > 0, clear_rows, lambda x: board, sub_board)

    return board, n_filled


def hard_drop(board: chex.Array, tetromino: chex.Array, x: int, y: int) -> tuple:
    """Performs a hard drop of the tetromino, moving it down as far as possible.

    Args:
        board: The current state of the Tetris board.
        tetromino: The tetromino to drop.
        x: The x-coordinate of the tetromino's position.
        y: The y-coordinate of the tetromino's position.

    Returns:
        The final y-coordinate after the hard drop.
    """

    def cond_fun(y):
        return ~collision(board, tetromino, x, y + 1)

    def body_fun(y):
        return y + 1

    new_y = jax.lax.while_loop(cond_fun, body_fun, y)
    reward = 2 * (new_y - y)  # 2 points per cell dropped
    return new_y, reward


def lock_active_tetromino(
    config: EnvConfig, tetrominoes: Tetrominoes, board, active_tetromino, rotation, x, y
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Locks the active tetromino in place, clears any filled rows, and calculates the reward.

    Args:
        config: Environment configuration.
        tetrominoes: Tetrominoes object containing tetromino configurations.
        board: The current state of the Tetris board.
        active_tetromino: ID of the active tetromino.
        rotation: Current rotation of the active tetromino.
        x: Current x-coordinate of the active tetromino.
        y: Current y-coordinate of the active tetromino.

    Returns:
        A tuple containing the updated board and the reward.
    """
    tetromino_matrix = get_tetromino_matrix(tetrominoes, active_tetromino, rotation)
    # place the tetromino on the board
    updated_board = project_tetromino(
        board, tetromino_matrix, x, y, tetrominoes.ids[active_tetromino]
    )
    # clear filled rows
    updated_board, lines_cleared = clear_filled_rows(config, tetrominoes, updated_board)
    # calculate reward
    reward = score(config, lines_cleared)

    return updated_board, reward, lines_cleared


def check_game_over(
    tetrominoes: Tetrominoes, board, active_tetromino, rotation, x, y
) -> bool:
    """Checks if the game is over by determining if the new tetromino collides immediately.

    Args:
        tetrominoes: Tetrominoes object containing tetromino configurations.
        board: The current state of the Tetris board.
        active_tetromino: ID of the active tetromino.
        rotation: Current rotation of the active tetromino.
        x: Current x-coordinate of the active tetromino.
        y: Current y-coordinate of the active tetromino.

    Returns:
        A boolean indicating whether the game is over.
    """
    return collision(
        board, get_tetromino_matrix(tetrominoes, active_tetromino, rotation), x, y
    )
