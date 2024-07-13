def test_observation_space_is_correct_after_reset(tetris_env):
    """Test that the observation space keys are correct after resetting the environment."""
    observation, info = tetris_env.reset(seed=42)
    assert tetris_env.observation_space.keys() == observation.keys()
