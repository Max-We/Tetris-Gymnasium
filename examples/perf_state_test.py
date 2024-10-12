import time
import numpy as np
import gymnasium as gym
from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.observation import RgbObservation
import copy


def create_env():
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    return RgbObservation(env)


def run_test(n_state_inits, n_copy_operations):
    # Create a single environment to use for all operations
    env = create_env()
    _ = env.reset()

    # Create n_state_inits copies of the state
    start_time = time.time()
    state_arr = [env.unwrapped.clone_state() for _ in range(n_state_inits)]
    creation_time = time.time() - start_time

    # Perform n_copy_operations
    start_time = time.time()
    for i in np.random.choice(n_state_inits, n_copy_operations, replace=False):
        state_arr[i] = env.unwrapped.clone_state()
    copy_operations_time = time.time() - start_time

    # Simulate a single copy operation
    start_time = time.time()
    idx = np.random.randint(n_state_inits)
    state = copy.deepcopy(env.unwrapped.clone_state())
    env.unwrapped.restore_state(state)
    state_arr[idx] = env.unwrapped.clone_state()
    single_copy_time = time.time() - start_time

    # Clean up
    env.close()

    return creation_time, copy_operations_time, single_copy_time


def main():
    n_state_inits_values = [500000]
    n_copy_operations_values = [500000]

    for n_state_inits in n_state_inits_values:
        for n_copy_operations in n_copy_operations_values:
            if n_copy_operations <= n_state_inits:
                creation_time, copy_operations_time, single_copy_time = run_test(n_state_inits, n_copy_operations)
                print(f"Test with {n_state_inits} state initializations and {n_copy_operations} copy operations")
                print(f"  State creation time: {creation_time:.6f} seconds")
                print(f"  Copy operations time: {copy_operations_time:.6f} seconds")
                print(f"  Single copy time: {single_copy_time:.6f} seconds")
                print(f"  Total time: {creation_time + copy_operations_time + single_copy_time:.6f} seconds")
                print()


if __name__ == "__main__":
    main()