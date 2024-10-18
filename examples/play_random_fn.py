from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax import random

from tetris_gymnasium.envs.tetris_fn import reset, step
from tetris_gymnasium.functional.logic import EnvConfig, State
from tetris_gymnasium.functional.tetrominoes import TETROMINOES, Tetrominoes


def print_board(board: chex.Array):
    """Print the Tetris board in a human-readable format."""
    for row in board:
        print(''.join(['.' if cell == 0 else '#' for cell in row]))
    print()

# Static constants
CONFIG = EnvConfig(width=10, height=20, padding=10, queue_size=3)
MAX_STEPS = 100

# Approach 1: Using static arguments
reset_static = jax.jit(reset, static_argnames=['config'])
step_static = jax.jit(step, static_argnames=['config'])
# reset_static = jax.jit(reset)
# step_static = jax.jit(step)
@jax.jit
def play_episode_static(key: chex.PRNGKey) -> Tuple[chex.PRNGKey, State, int]:
    """Play a single episode of Tetris with random actions (static constants)."""
    key, subkey = random.split(key)
    key, state = reset_static(TETROMINOES, subkey, CONFIG)

    def body_fun(carry):
        key, state, step_count = carry
        key, subkey = random.split(key)
        action = random.randint(subkey, (), 0, 7)  # 7 possible actions
        key, new_state, reward, game_over, info = step_static(TETROMINOES, key, state, action, CONFIG)
        return key, new_state, step_count + 1

    def cond_fun(carry):
        key, state, step_count = carry
        return (~state.game_over) & (step_count < MAX_STEPS)

    key, final_state, steps_taken = jax.lax.while_loop(cond_fun, body_fun, (key, state, 0))
    return key, final_state, steps_taken

# Approach 2: Using non-static arguments
reset_dynamic = jax.jit(reset)
step_dynamic = jax.jit(step)

@jax.jit
def play_episode_dynamic(key: chex.PRNGKey, const: Tetrominoes, config: EnvConfig) -> Tuple[chex.PRNGKey, State, int]:
    """Play a single episode of Tetris with random actions (dynamic constants)."""
    key, subkey = random.split(key)
    key, state = reset_dynamic(const, subkey, config)

    def body_fun(carry):
        key, state, step_count = carry
        key, subkey = random.split(key)
        action = random.randint(subkey, (), 0, 7)  # 7 possible actions
        key, new_state, reward, game_over, info = step_dynamic(const, key, state, action, config)
        return key, new_state, step_count + 1

    def cond_fun(carry):
        key, state, step_count = carry
        return (~state.game_over) & (step_count < MAX_STEPS)

    key, final_state, steps_taken = jax.lax.while_loop(cond_fun, body_fun, (key, state, 0))

    return key, final_state, steps_taken

# Main test function
def test_tetris_game():
    num_episodes = 3
    key = random.PRNGKey(0)

    print("Testing with static constants:")
    for episode in range(num_episodes):
        key, final_state, steps_taken = play_episode_static(key)
        print_board(final_state.board)
        print(f"Episode {episode + 1}: Steps: {steps_taken}, Score: {final_state.score}, Game over: {final_state.game_over}")

if __name__ == "__main__":
    # jax.config.update("jax_disable_jit", True)
    test_tetris_game()
