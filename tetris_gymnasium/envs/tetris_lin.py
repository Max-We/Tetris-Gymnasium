import numpy as np
from gymnasium.spaces import Box

from tetris_gymnasium.envs import Tetris


class TetrisLin(Tetris):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = Box(low=0, high=len(self.tetrominoes), shape=(self.height * self.width,), dtype=np.float32)

    def _get_obs(self):
        return self.board.reshape(-1).astype(np.float32)
