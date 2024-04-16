"""Observation wrapper module for the Tetris Gymnasium environment."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from tetris_gymnasium.envs import Tetris


class FlatObservation(gym.ObservationWrapper):
    """Wrapper that flattens the board observation."""

    def __init__(self, env: Tetris):
        """Initializes the observation space to be a 1-dimensional board observation."""
        super().__init__(env)
        self.observation_space = Box(
            low=0,
            high=len(env.tetrominoes),
            shape=(env.height * env.width,),
            dtype=np.float32,
        )

    def observation(self, observation):
        """Returns a 1-dimensional board observation."""
        # Todo: Decide if using `reshape(-1)` is better than `ravel()` or `flatten()`
        return self.board.reshape(-1).astype(np.float32)
