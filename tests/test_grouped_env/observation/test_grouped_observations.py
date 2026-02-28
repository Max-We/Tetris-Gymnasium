import numpy as np


def test_observation_space_is_correct_after_reset(tetris_env_grouped):
    """Test that the observation space is correct after resetting the environment."""
    assert tetris_env_grouped.observation_space.shape == (40, 24, 18)


def test_observation_space_is_correct_after_reset_with_other_wrappers(
    tetris_env_grouped_wrappers,
):
    """Test that the observation space is correct after resetting the environment."""
    assert tetris_env_grouped_wrappers.observation_space.shape == (40, 13)


def test_observation_index_is_correct(
    tetris_env_grouped, base_observation, expected_result_i_placement
):
    """Test that the observations in the observation space are indexed correctly."""
    observation = tetris_env_grouped.observation(base_observation)
    assert observation.shape == (40, 24, 18)

    assert np.all(
        observation[5 * 4 + 1] == expected_result_i_placement
    )  # expected: column (index) 3, rotation 1


def test_illegal_placement_observation_is_all_ones(
    tetris_env_grouped, base_observation
):
    """Test that illegal placements produce all-ones observations."""
    observation = tetris_env_grouped.observation(base_observation)

    # Find illegal actions
    illegal_actions = np.where(tetris_env_grouped.legal_actions_mask == 0)[0]
    assert len(illegal_actions) > 0, "Need at least one illegal action"

    for action in illegal_actions:
        assert np.all(
            observation[action] == 1
        ), f"Illegal action {action} should have all-ones observation"


def test_game_over_placement_observation_is_all_zeros(tetris_env_grouped):
    """Test that game-over placements produce all-zeros observations."""
    from tests.helpers.mock import convert_to_base_observation

    # Fill the board almost completely to trigger game-over placements
    board = np.copy(tetris_env_grouped.unwrapped.board)
    board[
        0 : tetris_env_grouped.unwrapped.height,
        tetris_env_grouped.unwrapped.padding : -tetris_env_grouped.unwrapped.padding,
    ] = 2
    tetris_env_grouped.unwrapped.board = board

    observation = tetris_env_grouped.observation(convert_to_base_observation(board))

    # Some actions should result in game-over (all-zeros) because the board is full
    has_zero_obs = False
    for i in range(40):
        if tetris_env_grouped.legal_actions_mask[i] == 1 and np.all(
            observation[i] == 0
        ):
            has_zero_obs = True
            break

    assert (
        has_zero_obs
    ), "Expected at least one game-over placement observation (all zeros)"
