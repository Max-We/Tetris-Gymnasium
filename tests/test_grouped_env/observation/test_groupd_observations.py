import numpy as np

from tests.helpers.mock import generate_example_board_with_features, convert_to_base_observation


def test_observation_space_is_correct_after_reset(tetris_env_grouped):
    """Test that the observation space is correct after resetting the environment."""
    assert tetris_env_grouped.observation_space.shape == (40, 24, 18)

def test_observation_space_is_correct_after_reset_with_other_wrappers(tetris_env_grouped_wrappers):
    """Test that the observation space is correct after resetting the environment."""
    assert tetris_env_grouped_wrappers.observation_space.shape == (40, 10)


def test_observation_index_is_correct(tetris_env_grouped, base_observation, expected_result_i_placement):
    """Test that the observation space is correct after resetting the environment."""
    observation = tetris_env_grouped.observation(base_observation)
    assert observation.shape == (40, 24, 18)

    assert np.all(observation[13] == expected_result_i_placement) # expected: column (index) 3, rotation 1
