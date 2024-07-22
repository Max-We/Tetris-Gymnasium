def test_actions_space_is_correct_after_reset(tetris_env):
    """Test that the action space is correct after resetting the environment."""
    tetris_env.reset(seed=42)
    assert (
        tetris_env.action_space.n == 8
    )  # hard coded test to remind update documentation after changing action space
