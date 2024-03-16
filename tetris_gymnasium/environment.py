from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium import Env
from gymnasium.core import RenderFrame, ActType, ObsType
from gymnasium.spaces import Discrete, Box
from gymnasium.vector.utils import spaces


class Tetris(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, width=10, height=20):
        self.width = width
        self.height = height
        self.window_width = width * 100
        self.window_height = height * 100

        self.observation_space = Box(low=0, high=1, shape=(width, height), dtype=np.int8)

        # rotate left / right, move left / right
        self.action_space = Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        pass

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        ObsType, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def close(self):
        super().close()

    @property
    def unwrapped(self) -> Env[ObsType, ActType]:
        return super().unwrapped

    @property
    def np_random(self) -> np.random.Generator:
        return super().np_random

    def __str__(self):
        return super().__str__()

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, *args: Any):
        return super().__exit__(*args)

    def get_wrapper_attr(self, name: str) -> Any:
        return super().get_wrapper_attr(name)
