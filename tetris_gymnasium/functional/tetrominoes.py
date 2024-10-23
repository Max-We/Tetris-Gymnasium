"""Contains Tetromino configurations and related functions."""
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp


class TetrominoType(NamedTuple):
    """Represents a type of Tetromino.

    Attributes:
        id (int): Unique identifier for the Tetromino.
        color (Tuple[int, int, int]): RGB color values for the Tetromino.
        matrix (chex.Array): 2D array representing the Tetromino's shape.
    """

    id: int
    color: Tuple[int, int, int]
    matrix: chex.Array


@chex.dataclass(frozen=True)
class Tetrominoes:
    """Contains all Tetromino configurations and related data.

    Attributes:
        base_pixels (chex.Array): Array of base pixel values.
        base_pixel_colors (chex.Array): Array of base pixel colors.
        ids (chex.Array): Array of Tetromino IDs.
        colors (chex.Array): Array of Tetromino colors.
        matrices (chex.Array): 3D array of Tetromino matrices for all rotations.
    """

    base_pixels: chex.Array
    base_pixel_colors: chex.Array
    ids: chex.Array
    colors: chex.Array
    matrices: chex.Array


# Define tetrominoes without padding
base_tetrominoes = (
    TetrominoType(
        id=2, color=(0, 240, 240), matrix=jnp.array([[1, 1, 1, 1]], dtype=jnp.uint8)
    ),  # I
    TetrominoType(
        id=3, color=(240, 240, 0), matrix=jnp.array([[1, 1], [1, 1]], dtype=jnp.uint8)
    ),  # O
    TetrominoType(
        id=4,
        color=(160, 0, 240),
        matrix=jnp.array([[0, 1, 0], [1, 1, 1]], dtype=jnp.uint8),
    ),  # T
    TetrominoType(
        id=5,
        color=(0, 240, 0),
        matrix=jnp.array([[0, 1, 1], [1, 1, 0]], dtype=jnp.uint8),
    ),  # S
    TetrominoType(
        id=6,
        color=(240, 0, 0),
        matrix=jnp.array([[1, 1, 0], [0, 1, 1]], dtype=jnp.uint8),
    ),  # Z
    TetrominoType(
        id=7,
        color=(0, 0, 240),
        matrix=jnp.array([[1, 0, 0], [1, 1, 1]], dtype=jnp.uint8),
    ),  # J
    TetrominoType(
        id=8,
        color=(240, 160, 0),
        matrix=jnp.array([[0, 0, 1], [1, 1, 1]], dtype=jnp.uint8),
    ),  # L
)


def rotate_90(matrix):
    """Rotate a matrix 90 degrees clockwise.

    Args:
        matrix (chex.Array): The input matrix to rotate.

    Returns:
        chex.Array: The rotated matrix.
    """
    return jnp.rot90(matrix, k=1)


def generate_rotations(matrix) -> "list[chex.Array]":
    """Generate all four rotations of a given matrix.

    Args:
        matrix (chex.Array): The input matrix to rotate.

    Returns:
        list[chex.Array]: A list containing the original matrix and its three rotations.
    """
    rotations = [matrix]
    for _ in range(3):
        rotations.append(rotate_90(rotations[-1]))
    return rotations


def pad_shape(matrix: chex.Array, max_size: int):
    """Pad a matrix to a specified maximum size.

    Args:
        matrix (chex.Array): The input matrix to pad.
        max_size (int): The desired size after padding.

    Returns:
        chex.Array: The padded matrix.
    """
    pad_width = [(0, max_size - matrix.shape[0]), (0, max_size - matrix.shape[1])]
    return jnp.pad(matrix, pad_width, mode="constant", constant_values=0)


# Generate all rotations
all_rotations = [generate_rotations(t.matrix) for t in base_tetrominoes]

# Find the maximum size needed for padding
max_size = max(max(t.matrix.shape[0], t.matrix.shape[1]) for t in base_tetrominoes)

# Pad all rotations to the maximum size
padded_rotations = [
    [pad_shape(rot, max_size) for rot in tetromino_rots]
    for tetromino_rots in all_rotations
]

# Define base pixels
base_pixels = jnp.array([0, 1], dtype=jnp.uint8)
base_pixels_colors = jnp.array(
    [[0, 0, 0], [128, 128, 128]], dtype=jnp.uint8  # Empty  # Bedrock
)

# Create the tetrominoes object
TETROMINOES = Tetrominoes(
    base_pixels=base_pixels,
    base_pixel_colors=base_pixels_colors,
    ids=jnp.array([t.id for t in base_tetrominoes], dtype=jnp.uint8),
    colors=jnp.array([t.color for t in base_tetrominoes], dtype=jnp.uint8),
    matrices=jnp.array(padded_rotations, dtype=jnp.uint8),
)


@jax.jit
def get_tetromino_matrix(
    tetrominoes: Tetrominoes, tetromino_id: int, rotation: int
) -> chex.Array:
    """Get the matrix for a specific Tetromino and rotation.

    Args:
        tetrominoes (Tetrominoes): The Tetrominoes object containing all configurations.
        tetromino_id (int): The ID of the desired Tetromino.
        rotation (int): The desired rotation (0-3).

    Returns:
        chex.Array: The matrix representing the Tetromino in the specified rotation.
    """
    return tetrominoes.matrices[tetromino_id, rotation]


@jax.jit
def get_tetromino_color(tetrominoes: Tetrominoes, tetromino_id: int) -> chex.Array:
    """Get the color for a specific Tetromino.

    Args:
        tetrominoes (Tetrominoes): The Tetrominoes object containing all configurations.
        tetromino_id (int): The ID of the desired Tetromino.

    Returns:
        chex.Array: The RGB color values for the specified Tetromino.
    """
    return jax.lax.dynamic_slice(tetrominoes.colors, (tetromino_id, 0), (1, 3))[0]
