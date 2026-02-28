import time

import gymnasium as gym
import numpy as np
import pytest

from tetris_gymnasium.components.tetromino_queue import TetrominoQueue
from tetris_gymnasium.components.tetromino_randomizer import Randomizer
from tetris_gymnasium.envs.tetris import TetrisState
from tetris_gymnasium.wrappers.observation import RgbObservation


def test_observation_space_is_correct_after_reset(tetris_env):
    """Test that the observation space keys are correct after resetting the environment."""
    observation, info = tetris_env.reset(seed=42)
    assert tetris_env.observation_space.keys() == observation.keys()


def test_observation_space_contains_all_expected_keys(tetris_env):
    """Test that observation contains board, active_tetromino_mask, holder, queue."""
    observation, _ = tetris_env.reset(seed=42)
    expected_keys = {"board", "active_tetromino_mask", "holder", "queue"}
    assert set(observation.keys()) == expected_keys


def test_board_dimensions_match_config(tetris_env):
    """Test that padded board shape matches height_padded x width_padded."""
    observation, _ = tetris_env.reset(seed=42)
    expected_shape = (
        tetris_env.unwrapped.height_padded,
        tetris_env.unwrapped.width_padded,
    )
    assert observation["board"].shape == expected_shape


def test_gravity_disabled():
    """Test that piece doesn't fall on no_op when gravity is disabled."""
    from tetris_gymnasium.mappings.actions import ActionsMapping

    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi", gravity=False)
    env.reset(seed=42)
    original_y = env.unwrapped.y

    env.step(ActionsMapping.no_op)

    assert env.unwrapped.y == original_y, "Piece should not move with gravity disabled"
    env.close()


def create_env():
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    return RgbObservation(env)


def compare_states(state1: TetrisState, state2: TetrisState):
    """Compare two Tetris game states."""
    assert (
        state1.__dict__.keys() == state2.__dict__.keys()
    ), "States have different attributes"

    for key in state1.__dict__.keys():
        if key == "board":
            assert np.array_equal(state1.board, state2.board), "Board mismatch"
        elif key == "active_tetromino":
            compare_tetrominos(state1.active_tetromino, state2.active_tetromino)
        elif key in ["x", "y", "has_swapped", "game_over", "score"]:
            assert getattr(state1, key) == getattr(
                state2, key
            ), f"Value mismatch for attribute: {key}"
        elif key == "queue":
            compare_tetromino_queues(state1.queue, state2.queue)
        elif key == "holder":
            compare_holders(state1.holder, state2.holder)
        elif key == "randomizer":
            compare_randomizers(state1.randomizer, state2.randomizer)
        else:
            raise ValueError(f"Unknown attribute in state: {key}")


def compare_tetrominos(tetromino1, tetromino2):
    """Compare two Tetromino objects."""
    assert tetromino1.id == tetromino2.id, "Tetromino ID mismatch"
    assert np.array_equal(
        tetromino1.color_rgb, tetromino2.color_rgb
    ), "Tetromino color mismatch"
    assert np.array_equal(
        tetromino1.matrix, tetromino2.matrix
    ), "Tetromino matrix mismatch"


def compare_tetromino_queues(queue1, queue2):
    """Compare two TetrominoQueue objects."""
    assert isinstance(queue1, TetrominoQueue) and isinstance(
        queue2, TetrominoQueue
    ), "Queue type mismatch"
    assert queue1.size == queue2.size, "Queue size mismatch"
    assert len(queue1.queue) == len(queue2.queue), "Queue length mismatch"
    for t1, t2 in zip(queue1.queue, queue2.queue):
        assert t1 == t2, "Queue content mismatch"
    compare_randomizers(queue1.randomizer, queue2.randomizer)


def compare_holders(holder1, holder2):
    """Compare two Holder objects."""
    assert holder1.size == holder2.size, "Holder size mismatch"
    assert len(holder1.queue) == len(holder2.queue), "Holder queue length mismatch"
    for t1, t2 in zip(holder1.queue, holder2.queue):
        if t1 is None and t2 is None:
            continue
        compare_tetrominos(t1, t2)


def compare_randomizers(randomizer1, randomizer2):
    """Compare two Randomizer objects."""
    assert isinstance(randomizer1, Randomizer) and isinstance(
        randomizer2, Randomizer
    ), "Randomizer type mismatch"
    assert randomizer1.__class__ == randomizer2.__class__, "Randomizer class mismatch"
    assert randomizer1.size == randomizer2.size, "Randomizer size mismatch"
    if hasattr(randomizer1, "bag"):
        assert np.array_equal(
            randomizer1.bag, randomizer2.bag
        ), "Randomizer bag mismatch"
        assert randomizer1.index == randomizer2.index, "Randomizer index mismatch"


# The compare_tetrominos, compare_tetromino_queues, compare_holders, and compare_randomizers
# functions remain the same as they deal with the internal objects, not the state structure itself


@pytest.fixture(scope="module")
def env():
    """Create a Tetris environment for testing."""
    environment = create_env()
    yield environment
    environment.close()


@pytest.mark.parametrize("test_number", range(100))
def test_clone_restore_consistency(env, test_number):
    """Test that cloning and restoring the environment state is consistent."""
    # Reset the environment if it's the first test or if the previous test ended
    if test_number == 0 or env.unwrapped.game_over:
        env.reset()

    # Clone the current state
    original_state: TetrisState = env.unwrapped.get_state()

    # Take a random action in the environment
    action = env.action_space.sample()
    original_obs, original_reward, original_done, original_info, _ = env.step(action)

    # Store the current state after the action
    post_action_state_a = env.unwrapped.get_state()

    # Restore the original state
    env.unwrapped.set_state(original_state)

    # Take the same action again
    cloned_obs, cloned_reward, cloned_done, cloned_info, _ = env.step(action)

    # Clone the current state
    post_action_state_b = env.unwrapped.get_state()

    # Compare results
    assert np.array_equal(original_obs, cloned_obs), "Observations don't match"
    assert original_reward == cloned_reward, "Rewards don't match"
    assert original_done == cloned_done, "Done flags don't match"
    assert original_info == cloned_info, "Info dictionaries don't match"

    # Compare the full state after action with the stored post-action state
    compare_states(post_action_state_a, post_action_state_b)


@pytest.mark.skip(reason="Performance test, run manually")
def test_state_copy_performance():
    """Test the performance of copying the state."""

    def run_test(n_state_inits, n_copy_operations):
        """Run a single test with the given number of state initializations and copy operations."""
        # Create a single environment to use for all operations
        env = create_env()
        _ = env.reset()

        # Create n_state_inits copies of the state
        start_time = time.time()
        state_arr = [env.unwrapped.get_state() for _ in range(n_state_inits)]
        creation_time = time.time() - start_time

        # Perform n_copy_operations
        start_time = time.time()
        for i in np.random.choice(n_state_inits, n_copy_operations, replace=False):
            state_arr[i] = env.unwrapped.get_state()
        copy_operations_time = time.time() - start_time

        # Simulate a single copy operation
        start_time = time.time()
        idx = np.random.randint(n_state_inits)
        state = env.unwrapped.get_state()
        env.unwrapped.set_state(state)
        state_arr[idx] = env.unwrapped.get_state()
        single_copy_time = time.time() - start_time

        # Clean up
        env.close()

        return creation_time, copy_operations_time, single_copy_time

    n_state_inits_values = [100000]
    n_copy_operations_values = [100000]

    for n_state_inits in n_state_inits_values:
        for n_copy_operations in n_copy_operations_values:
            if n_copy_operations <= n_state_inits:
                creation_time, copy_operations_time, single_copy_time = run_test(
                    n_state_inits, n_copy_operations
                )
                print(
                    f"Test with {n_state_inits} state initializations and {n_copy_operations} copy operations"
                )
                print(f"  State creation time: {creation_time:.6f} seconds")
                print(f"  Copy operations time: {copy_operations_time:.6f} seconds")
                print(f"  Single copy time: {single_copy_time:.6f} seconds")
                print(
                    f"  Total time: {creation_time + copy_operations_time + single_copy_time:.6f} seconds"
                )
                print()

    # Add an assertion to make pytest happy
    assert True, "Performance test completed"
