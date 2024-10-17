import jax
import jax.numpy as jnp
import chex
from typing import Tuple

@chex.dataclass
class TetrominoType:
    id: int
    color: Tuple[int, int, int]
    shape: chex.Array

@chex.dataclass(frozen=True)
class TetrisConstants:
    base_pixels: chex.Array
    base_pixel_colors: chex.Array
    tetromino_ids: chex.Array
    tetromino_colors: chex.Array
    tetromino_matrices: chex.Array
    all_colors: chex.Array

# Define base pixels
BASE_PIXELS = jnp.array([0, 1], dtype=jnp.uint8)
BASE_PIXEL_COLORS = jnp.array([
    [0, 0, 0],      # Empty
    [128, 128, 128] # Bedrock
], dtype=jnp.uint8)

# Define tetrominoes without padding
TETROMINOES = (
    TetrominoType(id=2, color=(0, 240, 240), shape=jnp.array([[1, 1, 1, 1]], dtype=jnp.uint8)),  # I
    TetrominoType(id=3, color=(240, 240, 0), shape=jnp.array([[1, 1], [1, 1]], dtype=jnp.uint8)),  # O
    TetrominoType(id=4, color=(160, 0, 240), shape=jnp.array([[0, 1, 0], [1, 1, 1]], dtype=jnp.uint8)),  # T
    TetrominoType(id=5, color=(0, 240, 0), shape=jnp.array([[0, 1, 1], [1, 1, 0]], dtype=jnp.uint8)),  # S
    TetrominoType(id=6, color=(240, 0, 0), shape=jnp.array([[1, 1, 0], [0, 1, 1]], dtype=jnp.uint8)),  # Z
    TetrominoType(id=7, color=(0, 0, 240), shape=jnp.array([[1, 0, 0], [1, 1, 1]], dtype=jnp.uint8)),  # J
    TetrominoType(id=8, color=(240, 160, 0), shape=jnp.array([[0, 0, 1], [1, 1, 1]], dtype=jnp.uint8)),  # L
)

def rotate_90(matrix):
    return jnp.rot90(matrix, k=1)

def generate_rotations(shape):
    rotations = [shape]
    for _ in range(3):
        rotations.append(rotate_90(rotations[-1]))
    return rotations

def pad_shape(shape, max_size):
    pad_width = [(0, max_size - shape.shape[0]), (0, max_size - shape.shape[1])]
    return jnp.pad(shape, pad_width, mode='constant', constant_values=0)

# Generate all rotations
all_rotations = [generate_rotations(t.shape) for t in TETROMINOES]

# Find the maximum size needed for padding
max_size = max(max(max(rot.shape[0], rot.shape[1]) for rot in tetromino_rots)
               for tetromino_rots in all_rotations)

# Pad all rotations to the maximum size
padded_rotations = [[pad_shape(rot, max_size) for rot in tetromino_rots]
                    for tetromino_rots in all_rotations]

# Create arrays for efficient access
TETROMINO_IDS = jnp.array([t.id for t in TETROMINOES], dtype=jnp.uint8)
TETROMINO_COLORS = jnp.array([t.color for t in TETROMINOES], dtype=jnp.uint8)
TETROMINO_MATRICES = jnp.array(padded_rotations, dtype=jnp.uint8)

# Combined color array for all pixel types
ALL_COLORS = jnp.vstack([BASE_PIXEL_COLORS, TETROMINO_COLORS]).astype(jnp.uint8)

# Create the constants object
TETRIS_CONSTANTS = TetrisConstants(
    base_pixels=BASE_PIXELS,
    base_pixel_colors=BASE_PIXEL_COLORS,
    tetromino_ids=TETROMINO_IDS,
    tetromino_colors=TETROMINO_COLORS,
    tetromino_matrices=TETROMINO_MATRICES,
    all_colors=ALL_COLORS
)

# Helper functions for JIT-compatible access
@jax.jit
def get_tetromino_matrix(constants: TetrisConstants, tetromino_index: int, rotation: int) -> chex.Array:
    return constants.tetromino_matrices[tetromino_index, rotation]

@jax.jit
def get_tetromino_color(constants: TetrisConstants, tetromino_index: int) -> chex.Array:
    return jax.lax.dynamic_slice(constants.tetromino_colors, (tetromino_index, 0), (1, 3))[0]