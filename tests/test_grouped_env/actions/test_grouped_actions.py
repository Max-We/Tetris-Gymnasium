import numpy as np


def test_action_mapping_is_correct(tetris_env_grouped, expected_result_i_placement):
    """Test that action-keys are correctly mapped by the environment"""
    tetris_env_grouped.step(5 * 4 + 1)  # expected: column (index) 5, rotation 1

    assert np.all(tetris_env_grouped.unwrapped.board == expected_result_i_placement)


def test_legal_action_mask_is_correct(tetris_env_grouped, base_observation):
    """Test that the legal action mask masks out invalid actions correctly"""
    # Generate action mask
    _ = tetris_env_grouped.observation(base_observation)

    print(tetris_env_grouped.legal_actions_mask.reshape((4, 10), order="F"))

    expected_action_mask = np.array(
        [
            [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    expected_action_mask = expected_action_mask.reshape(40, order="F")

    assert np.all(
        tetris_env_grouped.legal_actions_mask.astype(np.uint8)
        == expected_action_mask.astype(np.uint8)
    )  # expected: column (index) 3, rotation 1
