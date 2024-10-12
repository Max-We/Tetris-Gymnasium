import numpy as np
import pytest
import gymnasium as gym
from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.observation import RgbObservation
from tetris_gymnasium.components.tetromino_queue import TetrominoQueue
from tetris_gymnasium.components.tetromino_randomizer import Randomizer

def create_env():
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    return RgbObservation(env)

def compare_states(state1, state2):
    """Compare two Tetris game states."""
    assert state1.keys() == state2.keys(), "States have different keys"

    for key in state1.keys():
        if key == "board":
            assert np.array_equal(state1[key], state2[key]), f"Board mismatch"
        elif key == "active_tetromino":
            compare_tetrominos(state1[key], state2[key])
        elif key in ["x", "y", "has_swapped", "game_over", "score"]:
            assert state1[key] == state2[key], f"Value mismatch for key: {key}"
        elif key == "queue":
            compare_tetromino_queues(state1[key], state2[key])
        elif key == "holder":
            compare_holders(state1[key], state2[key])
        elif key == "randomizer":
            compare_randomizers(state1[key], state2[key])
        else:
            raise ValueError(f"Unknown key in state: {key}")

def compare_tetrominos(tetromino1, tetromino2):
    assert tetromino1.id == tetromino2.id, "Tetromino ID mismatch"
    assert np.array_equal(tetromino1.color_rgb, tetromino2.color_rgb), "Tetromino color mismatch"
    assert np.array_equal(tetromino1.matrix, tetromino2.matrix), "Tetromino matrix mismatch"

def compare_tetromino_queues(queue1, queue2):
    assert isinstance(queue1, TetrominoQueue) and isinstance(queue2, TetrominoQueue), "Queue type mismatch"
    assert queue1.size == queue2.size, "Queue size mismatch"
    assert len(queue1.queue) == len(queue2.queue), "Queue length mismatch"
    for t1, t2 in zip(queue1.queue, queue2.queue):
        assert t1 == t2, "Queue content mismatch"
    compare_randomizers(queue1.randomizer, queue2.randomizer)

def compare_holders(holder1, holder2):
    assert holder1.size == holder2.size, "Holder size mismatch"
    assert len(holder1.queue) == len(holder2.queue), "Holder queue length mismatch"
    for t1, t2 in zip(holder1.queue, holder2.queue):
        if t1 is None and t2 is None:
            continue
        compare_tetrominos(t1,t2)

def compare_randomizers(randomizer1, randomizer2):
    assert isinstance(randomizer1, Randomizer) and isinstance(randomizer2, Randomizer), "Randomizer type mismatch"
    assert randomizer1.__class__ == randomizer2.__class__, "Randomizer class mismatch"
    assert randomizer1.size == randomizer2.size, "Randomizer size mismatch"
    if hasattr(randomizer1, 'bag'):
        assert np.array_equal(randomizer1.bag, randomizer2.bag), "Randomizer bag mismatch"
        assert randomizer1.index == randomizer2.index, "Randomizer index mismatch"

@pytest.fixture(scope="module")
def env():
    environment = create_env()
    yield environment
    environment.close()

@pytest.mark.parametrize("test_number", range(1000))
def test_clone_restore_consistency(env, test_number):
    # Reset the environment if it's the first test or if the previous test ended
    if test_number == 0 or env.unwrapped.game_over:
        env.reset()

    # Clone the current state
    original_state = env.unwrapped.clone_state()

    # Take a random action in the environment
    action = env.action_space.sample()
    original_obs, original_reward, original_done, original_info, _ = env.step(action)

    # Store the current state after the action
    post_action_state_a = env.unwrapped.clone_state()

    # Restore the original state
    env.unwrapped.restore_state(original_state)

    # Take the same action again
    cloned_obs, cloned_reward, cloned_done, cloned_info, _ = env.step(action)

    # Clone the current state
    post_action_state_b = env.unwrapped.clone_state()

    # Compare results
    assert np.array_equal(original_obs, cloned_obs), "Observations don't match"
    assert original_reward == cloned_reward, "Rewards don't match"
    assert original_done == cloned_done, "Done flags don't match"
    assert original_info == cloned_info, "Info dictionaries don't match"

    # Compare the full state after action with the stored post-action state
    compare_states(post_action_state_a, post_action_state_b)
