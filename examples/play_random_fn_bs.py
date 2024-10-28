from typing import Tuple
import chex
import jax
import jax.numpy as jnp
from jax import random

from tetris_gymnasium.envs.tetris_fn import reset, step, batched_reset, batched_step
from tetris_gymnasium.functional.core import EnvConfig, State
from tetris_gymnasium.functional.tetrominoes import TETROMINOES, Tetrominoes


def print_board(board: chex.Array):
    """Print the Tetris board in a human-readable format."""
    for row in board:
        print(''.join(['.' if cell == 0 else '#' for cell in row]))
    print()


# Static constants
CONFIG = EnvConfig(width=10, height=20, padding=10, queue_size=3)
MAX_STEPS = 100
BATCH_SIZE = 32


@jax.jit
def play_parallel_episodes(rng: chex.PRNGKey) -> Tuple[chex.PRNGKey, State, chex.Array]:
    """Play multiple episodes of Tetris in parallel with random actions."""
    # Create separate keys for reset and main loop
    rng, reset_rng, loop_rng = random.split(rng, 3)

    # Split the reset key into batch size for reset
    reset_keys = random.split(reset_rng, BATCH_SIZE)

    # Reset all environments
    keys, states = batched_reset(
        TETROMINOES,
        reset_keys,
        config=CONFIG,
        batch_size=BATCH_SIZE
    )

    # Initialize step counters for all environments
    step_counts = jnp.zeros(BATCH_SIZE, dtype=jnp.int32)

    def body_fun(carry):
        keys, states, step_counts, rng = carry

        # Generate actions using vmap
        rng, action_rng = random.split(rng)
        action_rngs = random.split(action_rng, BATCH_SIZE)
        actions = jax.vmap(lambda k: random.randint(k, (), 0, 7))(action_rngs)

        # Step all environments using the keys from the previous step
        new_keys, new_states, rewards, dones, _ = batched_step(
            TETROMINOES,
            keys,  # Already batched from previous step
            states,
            actions,
            config=CONFIG
        )

        # Increment step counts where games aren't over
        new_step_counts = step_counts + (~new_states.game_over).astype(jnp.int32)

        return new_keys, new_states, new_step_counts, rng

    def cond_fun(carry):
        keys, states, step_counts, rng = carry
        # Continue if any game is not over and hasn't reached max steps
        return jnp.any(~states.game_over) & jnp.any(step_counts < MAX_STEPS)

    final_keys, final_states, steps_taken, final_rng = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (keys, states, step_counts, loop_rng)
    )

    return final_rng, final_states, steps_taken


def test_batched_tetris_game():
    key = random.PRNGKey(0)

    print(f"Testing with batch size {BATCH_SIZE}:")
    key, final_states, steps_taken = play_parallel_episodes(key)

    # Print results for first few environments
    num_to_display = 3
    for i in range(num_to_display):
        print(f"\nEnvironment {i + 1}:")
        print_board(final_states.board[i])
        print(f"Steps: {steps_taken[i]}, Score: {final_states.score[i]}, "
              f"Game over: {final_states.game_over[i]}")

    # Print summary statistics
    print("\nBatch Statistics:")
    print(f"Average steps: {jnp.mean(steps_taken):.2f}")
    print(f"Average score: {jnp.mean(final_states.score):.2f}")
    print(f"Games completed: {jnp.sum(final_states.game_over)} / {BATCH_SIZE}")


if __name__ == "__main__":
    # jax.config.update("jax_disable_jit", True)
    test_batched_tetris_game()