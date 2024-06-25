"""Registrations for the Tetris environment.

The `register` function is used to register the Tetris environment with Gymnasium.
Once registered, the environment can be created using the `gym.make` function.
"""
from gymnasium import register

from tetris_gymnasium.envs.tetris import Tetris

register(
    id="tetris_gymnasium/Tetris",
    entry_point="tetris_gymnasium.envs:Tetris",
    # max_episode_steps=1000,
)
