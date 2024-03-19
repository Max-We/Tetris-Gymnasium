from gymnasium import register

from tetris_gymnasium.envs.tetris import Tetris

# Todo: Move somewhere else?
register(
    id='tetris_gymnasium/Tetris',
    entry_point='tetris_gymnasium.envs:Tetris',
    max_episode_steps=10,
)
