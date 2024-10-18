
from typing import Tuple

import jax

# TODO: Implement holder logic

def swap_holder(active_tetromino: int, holder: int, has_swapped: bool) -> Tuple[int, int, bool]:
    return jax.lax.cond(
        ~has_swapped,
        lambda: (holder if holder != -1 else active_tetromino, active_tetromino, True),
        lambda: (active_tetromino, holder, has_swapped)
    )
