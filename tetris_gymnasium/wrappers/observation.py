"""Observation wrapper module for the Tetris Gymnasium environment."""

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box

from tetris_gymnasium.envs import Tetris


class RgbObservation(gym.ObservationWrapper):
    """Observation wrapper that displays all observations (board, holder, queue) as one single RGB Image.

    The observation contains the board on the left, the queue on the top right and the holder on the bottom right.
    """

    def __init__(self, env: Tetris):
        """The size of the matrix depends on how many tetrominoes can be stored in the queue / holder."""
        super().__init__(env)
        self.observation_space = Box(
            low=0,
            high=len(env.unwrapped.tetrominoes),
            shape=(
                env.unwrapped.height_padded,
                env.unwrapped.width_padded
                + max(env.unwrapped.holder.size, env.unwrapped.queue.size)
                * env.unwrapped.padding,
                3,
            ),
            dtype=np.uint8,
        )

    def observation(self, observation):
        """Observation wrapper that displays all observations (board, holder, queue) as one single RGB Image.

        The observation contains the board on the left, the queue on the top right and the holder on the bottom right.
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

        stack = np.hstack((board_obs, cnn_extra)).astype(np.integer)

        # Convert to RGB
        rgb = np.zeros((stack.shape[0], stack.shape[1], 3))
        colors = np.array(
            list(p.color_rgb for p in self.env.unwrapped.pixels), dtype=np.uint8
        )
        rgb[...] = colors[stack]

        return rgb.astype(np.uint8)

    def render(self) -> "RenderFrame | list[RenderFrame] | None":
        """Renders the environment in various formats.

        This render function is different from the default as it uses the values from :func:`observation`  to render
        the environment.
        """
        matrix = self.observation(self.env.unwrapped._get_obs())

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            if self.render_mode == "rgb_array":
                return matrix

            if self.render_mode == "human":
                if self.env.unwrapped.window_name is None:
                    self.env.unwrapped.window_name = "Tetris Gymnasium"
                    cv2.namedWindow(
                        self.env.unwrapped.window_name, cv2.WINDOW_GUI_NORMAL
                    )
                    assert self.observation_space.shape is not None
                    h, w = (
                        self.observation_space.shape[0],
                        self.observation_space.shape[1],
                    )
                    cv2.resizeWindow(self.env.unwrapped.window_name, w * 10, h * 10)
                cv2.imshow(
                    self.env.unwrapped.window_name,
                    cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR),
                )

        return None


class GroupedActionRgbObservation(gym.ObservationWrapper):
    """Observation wrapper that displays all observations (board, holder, queue) as one single RGB Image.

    The observation contains the board on the left, the queue on the top right and the holder on the bottom right.
    """

    def __init__(self, env: Tetris):
        """The size of the matrix depends on how many tetrominoes can be stored in the queue / holder."""
        super().__init__(env)
        self.observation_space = Box(
            low=0,
            high=len(env.unwrapped.tetrominoes),
            shape=(
                env.unwrapped.height_padded * 4,
                (
                    env.unwrapped.width_padded
                    + max(env.unwrapped.holder.size, env.unwrapped.queue.size)
                    * env.unwrapped.padding
                )
                * env.unwrapped.width,
                3,
            ),
            dtype=np.uint8,
        )

    def observation(self, observation):
        """Observation wrapper that displays all observations (board, holder, queue) as one single RGB Image.

        The observation contains the board on the left, the queue on the top right and the holder on the bottom right.
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

        v_padding = np.ones((board_obs.shape[1] - 2 * max_size, max_len))
        cnn_extra = np.vstack((queue_obs, v_padding, holder_obs))

        # add vertical padding between the board and the holder/queue
        boards = np.array(
            [np.hstack((o, cnn_extra)).astype(np.integer) for o in board_obs]
        )

        # Create 2D representation of all boards
        n_rotations = 4
        rotated_groups = np.split(boards, boards.shape[0] // n_rotations)
        matrix = np.hstack([np.vstack(r) for r in rotated_groups])

        # Convert to rgb
        colors = np.array(
            list(p.color_rgb for p in self.env.unwrapped.pixels), dtype=np.uint8
        )
        matrix_rgb = colors[matrix]
        self.obs = matrix_rgb.astype(np.uint8)

        return self.obs

    def render(self) -> "RenderFrame | list[RenderFrame] | None":
        """Renders the environment in various formats.

        This render function is different from the default as it uses the values from :func:`observation`  to render
        the environment.
        """
        matrix = self.obs
        if matrix is None:
            return None

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            if self.render_mode == "rgb_array":
                return matrix

            if self.render_mode == "human":
                if self.env.unwrapped.window_name is None:
                    self.env.unwrapped.window_name = "Tetris Gymnasium"
                    cv2.namedWindow(
                        self.env.unwrapped.window_name, cv2.WINDOW_GUI_NORMAL
                    )
                    assert self.observation_space.shape is not None
                    h, w = (
                        self.observation_space.shape[0],
                        self.observation_space.shape[1],
                    )
                    cv2.resizeWindow(self.env.unwrapped.window_name, w * 2, h * 2)
                cv2.imshow(
                    self.env.unwrapped.window_name,
                    cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR),
                )

        return None
