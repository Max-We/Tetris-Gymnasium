"""Observation wrapper module for the Tetris Gymnasium environment."""
import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box

from tetris_gymnasium.envs import Tetris


class CnnObservation(gym.ObservationWrapper):
    """Wrapper that displays all observations (board, holder, queue) in a single 2D matrix.

    The 2D matrix contains the board on the left, the queue on the top right and the holder on the bottom right.
    """

    def __init__(self, env: Tetris):
        """Initializes the observation space to be a single 2D matrix.

        The size of the matrix depends on how many tetrominoes can be stored in the queue / holder.
        """
        super().__init__(env)
        self.observation_space = Box(
            low=0,
            high=len(env.unwrapped.tetrominoes),
            shape=(
                env.unwrapped.height_padded,
                env.unwrapped.width_padded
                + max(env.unwrapped.holder.size, env.unwrapped.queue.size)
                * env.unwrapped.padding,
            ),
            dtype=np.float32,
        )

    def observation(self, observation):
        """Wrapper that displays all observations (board, holder, queue) in a single 2D matrix.

        The 2D matrix contains the board on the left, the queue on the top right and the holder on the bottom right.
        """
        # Board
        board_obs = observation["board"]
        # Holder
        holder_obs = observation["holder"]
        # Queue
        queue_obs = observation["queue"]

        max_size = holder_obs.shape[0]
        max_len = max(holder_obs.shape[1], queue_obs.shape[1])

        # make holder and queue same length by adding optional padding
        holder_obs = np.hstack(
            (holder_obs, np.ones((max_size, max_len - holder_obs.shape[1])))
        )
        queue_obs = np.hstack(
            (queue_obs, np.ones((max_size, max_len - queue_obs.shape[1])))
        )

        # add vertical padding between the board and the holder/queue
        v_padding = np.ones((board_obs.shape[0] - 2 * max_size, max_len))
        cnn_extra = np.vstack((queue_obs, v_padding, holder_obs))

        return np.hstack((board_obs, cnn_extra))

    def render(self) -> "RenderFrame | list[RenderFrame] | None":
        """Renders the environment in various formats.

        This render function is different from the default as it uses the observation space to render the environment.
        """
        matrix = self.observation(self.env.unwrapped._get_obs()).astype(np.integer)

        if self.render_mode == "ansi":
            char_field = np.where(matrix == 0, ".", matrix.astype(str))
            field_str = "\n".join("".join(row) for row in char_field)
            return field_str
        if self.render_mode == "rgb_array":
            # Initialize rgb array
            rgb = np.zeros((matrix.shape[0], matrix.shape[1], 3), dtype=np.uint8)
            # Render the board
            colors = np.array(list(p.color_rgb for p in self.pixels), dtype=np.uint8)
            rgb[...] = colors[matrix]

            return rgb

        return None
