"""Observation wrapper module for the Tetris Gymnasium environment."""
import copy
from typing import Any

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

class FeatureVectorObservation(gym.ObservationWrapper):
    """Observation wrapper that returns the feature vector as the observation.

    The feature vector contains the board, the queue and the holder.
    """

    def __init__(self, env: Tetris):
        """The size of the matrix depends on how many tetrominoes can be stored in the queue / holder."""
        super().__init__(env)
        self.observation_space = Box(
            low=0,
            high=len(env.unwrapped.tetrominoes),
            shape=(
                (env.unwrapped.width + 1,)
            ),
            dtype=np.uint8,
        )

    def calculate_height(self, board):
        """Calculate the height of each column in the board."""
        # Slicing the board to remove padding
        board = board[
                0: -self.env.unwrapped.padding,
                self.env.unwrapped.padding: -self.env.unwrapped.padding,
                ]

        # Create a mask where board is not equal to 0
        mask = board != 0

        # Get the indices of the first non-zero element in each column
        height = np.argmax(mask, axis=0)

        # Columns with no blocks should have height 0
        height[np.all(mask == 0, axis=0)] = 0

        # For columns with blocks, calculate the height from the bottom of the board
        valid_heights = np.any(mask, axis=0)
        height[valid_heights] = board.shape[0] - height[valid_heights]

        return height

    def calculate_max_height(self, height):
        """Calculate the maximum height among all columns."""
        return np.max(height)

    def calculate_holes(self, board):
        """Calculate the number of holes in the board."""
        board = copy.deepcopy(board)
        board = board[
                0: -self.env.unwrapped.padding,
                self.env.unwrapped.padding: -self.env.unwrapped.padding,
                ]

        if np.all(board == np.ones_like(board)):
            return board.shape[0] * board.shape[1]

        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < len(board) and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board):
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(
            mask.any(axis=0), np.argmax(mask, axis=0), self.env.unwrapped.height
        )
        heights = self.env.unwrapped.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def observation(self, observation):
        """Observation wrapper that returns the feature vector as the observation.

        The feature vector contains the board, the queue and the holder.
        """
        # Board
        board_obs = observation["board"]

        height = self.calculate_height(board_obs)
        max_height = self.calculate_max_height(height)

        # holes_obs = np.array([self.calculate_holes(board) for board in grouped_board_obs])
        # hb_obs = np.array([self.get_bumpiness_and_height(board) for board in grouped_board_obs])

        observation["board"] = np.array([*height, max_height])
        return observation ["board"]