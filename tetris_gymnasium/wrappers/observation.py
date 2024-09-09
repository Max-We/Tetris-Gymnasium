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
    The size of the matrix depends on how many tetrominoes can be stored in the queue / holder.
    """

    def __init__(self, env: Tetris):
        """Initialize the RgbObservation wrapper.

        Args:
            env (Tetris): The environment
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

        stack = np.hstack((board_obs, cnn_extra)).astype(np.uint8)

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
        kernel = np.ones(
            (self.render_scaling_factor, self.render_scaling_factor, 1),
            dtype=np.uint8,
        )
        matrix = np.kron(matrix, kernel)

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
                cv2.waitKey(1)

        return None


class FeatureVectorObservation(gym.ObservationWrapper):
    """Observation wrapper that returns a feature vector as observation.

    **State representation**
        A feature vector can contain different features of the board, such as the height of the stack or the number of holes.
        In the literature, this is often referred to as a state representation and many different features can be used. A
        discussion about the state representation can be found in "Reinforcement learning (RL) is a paradigm within machine
        learning that has been applied to Tetris, demonstrating the effect of state representation on performance
        (Hendriks)."

    **Features**
        For this wrapper, the features from https://github.com/uvipen/Tetris-deep-Q-learning-pytorch have been
        adapted. These features are:

        - The height of the stack in each column (list: int for each column)
        - The maximum height of the stack (int)
        - The number of holes in the stack (int)
        - The bumpiness of the stack (int)

        More features can be added in the future or by introducing new wrappers.
    """

    def __init__(
        self,
        env: Tetris,
        report_height=True,
        report_max_height=True,
        report_holes=True,
        report_bumpiness=True,
    ):
        """Initialize the FeatureVectorObservation wrapper.

        Args:
            env (Tetris): The environment.
            report_height (bool, optional): Report the height of the stack in each column. Defaults to True.
            report_max_height (bool, optional): Report the maximum height of the stack. Defaults to True.
            report_holes (bool, optional): Report the number of holes in the stack. Defaults to True.
            report_bumpiness (bool, optional): Report the bumpiness of the stack. Defaults to True.
        """
        super().__init__(env)
        self.observation_space = Box(
            low=0,
            high=len(env.unwrapped.tetrominoes),
            shape=(
                (
                    (env.unwrapped.width if report_height else 0)
                    + (1 if report_max_height else 0)
                    + (1 if report_holes else 0)
                    + (1 if report_bumpiness else 0),
                )
            ),
            dtype=np.uint8,
        )

        self.report_height = report_height
        self.report_max_height = report_max_height
        self.report_holes = report_holes
        self.report_bumpiness = report_bumpiness

    def calc_height(self, board):
        """Calculate the height of the board.

        Args:
            board (np.ndarray): The board.

        Returns:
            np.ndarray: The height of the stack in each column.
        """
        # Find the lowest non-zero element in each column
        heights = board.shape[0] - np.argmax(
            board != 0, axis=0
        )  # measure top to bottom to avoid holes
        heights = np.where(
            np.all(board == 0, axis=0), 0, heights
        )  # empty columns should be 0 (not 20)
        return heights

    def calc_max_height(self, board):
        """Calculate the maximum height of the board.

        Args:
            board (np.ndarray): The board.

        Returns:
            int: The maximum height of the board.
        """
        # Find the maximum height across all columns
        return np.max(self.calc_height(board))

    def calc_bumpiness(self, board):
        """Calculate the bumpiness of the board.

        Bumpiness is the sum of the absolute differences between adjacent column heights.

        Args:
            board (np.ndarray): The board.

        Returns:
            int: The bumpiness of the board.
        """
        heights = self.calc_height(board)
        # Calculate differences between adjacent heights and sum their absolute values
        return np.sum(np.abs(np.diff(heights)))

    def calc_holes(self, board):
        """Calculate the number of holes in the stack.

        Args:
            board (np.ndarray): The board.

        Returns:
            int: The number of holes in the stack.
        """
        # Create a mask of non-zero elements
        filled = board != 0
        # Calculate cumulative sum of filled cells from top to bottom
        cumsum = np.cumsum(filled, axis=0)
        # Count cells that are empty but have filled cells above them
        return np.sum((board == 0) & (cumsum > 0))

    def observation(self, observation):
        """Observation wrapper that returns the feature vector as the observation.

        Args:
            observation (dict): The observation from the base environment.

        Returns:
            np.ndarray: The feature vector.
        """
        # Board
        board_obs = observation["board"]
        active_tetromino_mask = observation["active_tetromino_mask"]

        # mask out the active tetromino
        board_obs[active_tetromino_mask] = 0
        # crop the board to remove padding
        board_obs = board_obs[
            0 : -self.env.unwrapped.padding,
            self.env.unwrapped.padding : -self.env.unwrapped.padding,
        ]

        features = []

        if self.report_height or self.report_max_height:
            height_vector = self.calc_height(board_obs)
            if self.report_height:
                features += list(height_vector)
            if self.report_max_height:
                max_height = np.max(height_vector)
                features.append(max_height)

        if self.report_holes:
            holes = self.calc_holes(board_obs)
            features.append(holes)

        if self.report_bumpiness:
            bumpiness = self.calc_bumpiness(board_obs)
            features.append(bumpiness)

        features = np.array(features, dtype=np.uint8)
        return features
