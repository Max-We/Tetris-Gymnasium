import numpy as np


def test_action_mapping_is_correct(tetris_env_grouped, expected_result_i_placement):
    """Test that action-keys are correctly mapped by the environment"""
    tetris_env_grouped.step(13 + 4 * 3)  # expected: column (index) 3, rotation 1

    assert np.all(tetris_env_grouped.unwrapped.board == expected_result_i_placement)


def test_legal_action_mask_is_correct(tetris_env_grouped, base_observation):
    """Test that the legal action mask masks out invalid actions correctly"""
    # Generate action mask
    _ = tetris_env_grouped.observation(base_observation)

    print(tetris_env_grouped.legal_actions_mask)

    pad_offset = 3
    expected_action_mask = np.ones(52)
    expected_action_mask = expected_action_mask.reshape(4, 13, order="F")
    expected_action_mask[:, 0] = 0  # illegal because of padding
    expected_action_mask[
        [0, 1, 3], 1
    ] = 0  # illegal because of padding: all except I piece left vertical rotation
    expected_action_mask[
        [1, 3], 2:3
    ] = 0  # illegal because of padding: all horizontal rotations for I piece
    expected_action_mask[:, 9 + pad_offset] = 0
    expected_action_mask[[1, 3], 7 + pad_offset] = 0

    # the other vertical orientation (2) is also illegal because padding is larger on the left side
    expected_action_mask[[1, 2, 3], 8 + 3] = 0
    expected_action_mask = expected_action_mask.reshape(52, order="F")

    print(expected_action_mask)

    assert np.all(
        tetris_env_grouped.legal_actions_mask.astype(np.uint8)
        == expected_action_mask.astype(np.uint8)
    )  # expected: column (index) 3, rotation 1
