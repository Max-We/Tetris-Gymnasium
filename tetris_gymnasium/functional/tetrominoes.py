import jax
import jax.numpy as jnp
import chex
from typing import Tuple, NamedTuple


class TetrominoType(NamedTuple):
    id: int
    color: Tuple[int, int, int]
    matrix: chex.Array

@chex.dataclass(frozen=True)
class Tetrominoes:
    base_pixels: chex.Array
    base_pixel_colors: chex.Array
    ids: chex.Array
    colors: chex.Array
    matrices: chex.Array

# Define tetrominoes without padding
base_tetrominoes = (
    TetrominoType(id=2, color=(0, 240, 240), matrix=jnp.array([[1, 1, 1, 1]], dtype=jnp.uint8)),  # I
    TetrominoType(id=3, color=(240, 240, 0), matrix=jnp.array([[1, 1], [1, 1]], dtype=jnp.uint8)),  # O
    TetrominoType(id=4, color=(160, 0, 240), matrix=jnp.array([[0, 1, 0], [1, 1, 1]], dtype=jnp.uint8)),  # T
    TetrominoType(id=5, color=(0, 240, 0), matrix=jnp.array([[0, 1, 1], [1, 1, 0]], dtype=jnp.uint8)),  # S
    TetrominoType(id=6, color=(240, 0, 0), matrix=jnp.array([[1, 1, 0], [0, 1, 1]], dtype=jnp.uint8)),  # Z
    TetrominoType(id=7, color=(0, 0, 240), matrix=jnp.array([[1, 0, 0], [1, 1, 1]], dtype=jnp.uint8)),  # J
    TetrominoType(id=8, color=(240, 160, 0), matrix=jnp.array([[0, 0, 1], [1, 1, 1]], dtype=jnp.uint8)),  # L
)

def rotate_90(matrix):
    return jnp.rot90(matrix, k=1)

def generate_rotations(matrix) -> list[chex.Array]:
    rotations = [matrix]
    for _ in range(3):
        rotations.append(rotate_90(rotations[-1]))
    return rotations

def pad_shape(matrix: chex.Array, max_size: int):
    pad_width = [(0, max_size - matrix.shape[0]), (0, max_size - matrix.shape[1])]
    return jnp.pad(matrix, pad_width, mode='constant', constant_values=0)

# Generate all rotations
all_rotations = [generate_rotations(t.matrix) for t in base_tetrominoes]

# Find the maximum size needed for padding
max_size = max(max(t.matrix.shape[0], t.matrix.shape[1]) for t in base_tetrominoes)

# Pad all rotations to the maximum size
padded_rotations = [[pad_shape(rot, max_size) for rot in tetromino_rots]
                    for tetromino_rots in all_rotations]

# Define base pixels
base_pixels = jnp.array([0, 1], dtype=jnp.uint8)
base_pixels_colors = jnp.array([
    [0, 0, 0],      # Empty
    [128, 128, 128] # Bedrock
], dtype=jnp.uint8)

# Create the tetrominoes object
TETROMINOES = Tetrominoes(
    base_pixels=base_pixels,
    base_pixel_colors=base_pixels_colors,
    ids=jnp.array([t.id for t in base_tetrominoes], dtype=jnp.uint8),
    colors=jnp.array([t.color for t in base_tetrominoes], dtype=jnp.uint8),
    matrices=jnp.array(padded_rotations, dtype=jnp.uint8),
)

# Helper functions for JIT-compatible access
@jax.jit
def get_tetromino_matrix(tetrominoes: Tetrominoes, tetromino_id: int, rotation: int) -> chex.Array:
    return tetrominoes.matrices[tetromino_id, rotation]

@jax.jit
def get_tetromino_color(tetrominoes: Tetrominoes, tetromino_id: int) -> chex.Array:
    return jax.lax.dynamic_slice(tetrominoes.colors, (tetromino_id, 0), (1, 3))[0]
