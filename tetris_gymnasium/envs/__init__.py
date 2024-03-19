from gymnasium import register

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.envs.tetris_lin import TetrisLin

register(
    id='tetris_gymnasium/Tetris',
    entry_point='tetris_gymnasium.envs:Tetris',
    max_episode_steps=10,
)

register(
    id='tetris_gymnasium/TetrisLin',
    entry_point='tetris_gymnasium.envs:TetrisLin',
    max_episode_steps=10,
)
