import jax
import jax.numpy as jnp
from typing import NamedTuple

class TetrisConstants(NamedTuple):
    base_pixels: jnp.ndarray
    base_pixel_colors: jnp.ndarray
    tetromino_ids: jnp.ndarray
    tetromino_colors: jnp.ndarray
    tetromino_matrices: jnp.ndarray
    all_colors: jnp.ndarray

def function(x, static_params: TetrisConstants):
    # This is just an example function. You would replace this with your actual Tetris logic.
    return x + static_params.base_pixels.sum() + static_params.tetromino_ids.sum()

jitted_function = jax.jit(function, static_argnames=('static_params',))

# Example values for TetrisConstants
static_params = TetrisConstants(
    base_pixels=jnp.array([[1, 0], [0, 1]]),
    base_pixel_colors=jnp.array([0.1, 0.2, 0.3]),
    tetromino_ids=jnp.array([1, 2, 3, 4]),
    tetromino_colors=jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    tetromino_matrices=jnp.array([[[1, 0], [1, 1]], [[1, 1], [0, 1]]]),
    all_colors=jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
)

x = jnp.array([1.0, 2.0, 3.0])
result = jitted_function(x, static_params)
print(result)