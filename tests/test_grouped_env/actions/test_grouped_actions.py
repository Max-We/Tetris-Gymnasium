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


def test_encode_decode_roundtrip(tetris_env_grouped):
    """Test that encode(decode(a)) == a for all 40 actions."""
    for action in range(40):
        x, r = tetris_env_grouped.decode_action(action)
        assert tetris_env_grouped.encode_action(x, r) == action


def test_action_mask_updates_after_step(tetris_env_grouped):
    """Test that action mask changes after taking an action (new piece may have different legality)."""
    mask_before = tetris_env_grouped.legal_actions_mask.copy()

    # Find a legal action and take it
    legal_actions = np.where(mask_before == 1)[0]
    assert len(legal_actions) > 0
    obs, reward, terminated, truncated, info = tetris_env_grouped.step(legal_actions[0])

    if not terminated:
        # Mask should be present in info
        assert "action_mask" in info
        # New mask should exist (may or may not differ depending on piece)
        assert info["action_mask"].shape == (40,)


def test_all_legal_actions_produce_valid_boards(tetris_env_grouped):
    """Test that all legal actions produce valid results without crashing."""
    import copy

    import gymnasium as gym

    from tests.helpers.mock import (
        convert_to_base_observation,
        generate_example_board_with_features,
    )
    from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations

    legal_actions = np.where(tetris_env_grouped.legal_actions_mask == 1)[0]

    for action in legal_actions:
        # Create a fresh env for each action to avoid state contamination
        env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
        env = GroupedActionsObservations(env)
        env.reset(seed=42)

        example_board, _, _, _, _ = generate_example_board_with_features(env)
        env.unwrapped.board = example_board
        env.unwrapped.active_tetromino = copy.deepcopy(
            tetris_env_grouped.unwrapped.active_tetromino
        )
        _ = env.observation(convert_to_base_observation(example_board))

        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        env.close()
