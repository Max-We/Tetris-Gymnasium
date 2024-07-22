from tetris_gymnasium.mappings.rewards import RewardsMapping


def test_line_clear_normal(tetris_env_grouped, base_observation):
    """Test that the reward is given after clearing a line."""
    observation, reward, terminated, truncated, info = tetris_env_grouped.step(
        32
    )  # drop I-tetromino in the last column
    assert (
        reward >= RewardsMapping.clear_line
    )  # In the future, this reward formula may change
