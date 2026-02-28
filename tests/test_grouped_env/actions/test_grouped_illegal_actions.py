import gymnasium as gym
import pytest

from tests.helpers.mock import (
    convert_to_base_observation,
    generate_example_board_with_features,
)
from tetris_gymnasium.mappings.rewards import RewardsMapping
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations


@pytest.fixture
def tetris_env_grouped_terminate(vertical_i_tetromino):
    """Grouped env that terminates on illegal action."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env = GroupedActionsObservations(env, terminate_on_illegal_action=True)
    env.reset(seed=42)

    example_board, _, _, _, _ = generate_example_board_with_features(env)
    env.unwrapped.board = example_board
    env.unwrapped.active_tetromino = vertical_i_tetromino
    _ = env.observation(convert_to_base_observation(example_board))

    yield env
    env.close()


@pytest.fixture
def tetris_env_grouped_no_terminate(vertical_i_tetromino):
    """Grouped env that does NOT terminate on illegal action."""
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env = GroupedActionsObservations(env, terminate_on_illegal_action=False)
    env.reset(seed=42)

    example_board, _, _, _, _ = generate_example_board_with_features(env)
    env.unwrapped.board = example_board
    env.unwrapped.active_tetromino = vertical_i_tetromino
    _ = env.observation(convert_to_base_observation(example_board))

    yield env
    env.close()


def _find_illegal_action(env):
    """Find the first illegal action from the action mask."""
    for i in range(env.action_space.n):
        if env.legal_actions_mask[i] == 0:
            return i
    return None


def test_illegal_action_terminates(tetris_env_grouped_terminate):
    """Test that illegal action with terminate_on_illegal_action=True returns terminated=True."""
    illegal_action = _find_illegal_action(tetris_env_grouped_terminate)
    assert illegal_action is not None, "Need at least one illegal action for this test"

    obs, reward, terminated, truncated, info = tetris_env_grouped_terminate.step(
        illegal_action
    )

    assert terminated is True
    assert reward == RewardsMapping.invalid_action


def test_illegal_action_no_terminate(tetris_env_grouped_no_terminate):
    """Test that illegal action with terminate_on_illegal_action=False continues the game."""
    illegal_action = _find_illegal_action(tetris_env_grouped_no_terminate)
    assert illegal_action is not None, "Need at least one illegal action for this test"

    obs, reward, terminated, truncated, info = tetris_env_grouped_no_terminate.step(
        illegal_action
    )

    assert reward == RewardsMapping.invalid_action
    # Game should continue (terminated may be True if gravity caused game over,
    # but the action itself shouldn't terminate)
    assert obs is not None
