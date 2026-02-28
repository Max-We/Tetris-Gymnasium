from tetris_gymnasium.mappings.rewards import RewardsMapping


def test_line_clear_normal(tetris_env_grouped, base_observation):
    """Test that the reward is given after clearing a line."""
    observation, reward, terminated, truncated, info = tetris_env_grouped.step(
        32
    )  # drop I-tetromino in the last column
    assert (
        reward >= RewardsMapping.clear_line
    )  # In the future, this reward formula may change


def test_grouped_line_clear_reward_matches_base_env(tetris_env_grouped):
    """Test that grouped line clear reward matches base env score formula."""
    # The tetris_env_grouped fixture has the example board with a vertical I-tetromino
    # Action 32 drops I-tetromino in column 8 with rotation 0
    # which should clear lines based on the example board setup
    _, reward, terminated, _, info = tetris_env_grouped.step(32)

    if info["lines_cleared"] > 0:
        lines = info["lines_cleared"]
        expected_score = (lines**2) * tetris_env_grouped.unwrapped.width
        # Reward = score + alife
        assert reward == expected_score + RewardsMapping.alife
