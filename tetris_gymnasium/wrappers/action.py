"""Observation wrapper that groups the actions into placements.

The action space is the width of the board times 4 (4 rotations).
"""
import copy
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from tetris_gymnasium.components.tetromino import Tetromino
from tetris_gymnasium.envs import Tetris


class GroupedActions(gym.ObservationWrapper):
    """Observation wrapper that groups the actions into placements.

    The action space is the width of the board times 4 (4 rotations).
    """

    def __init__(self, env: Tetris):
        """Initializes the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self.action_space = Discrete(env.unwrapped.width * 4)
        self.observation_space = gym.spaces.Dict(
            {
                "board": Box(
                    low=0,
                    high=len(env.unwrapped.tetrominoes),
                    shape=(
                        env.unwrapped.width * 4,
                        env.unwrapped.height_padded,
                        env.unwrapped.width_padded,
                    ),
                    dtype=np.uint8,
                ),
                "holder": Box(
                    low=0,
                    high=len(self.env.unwrapped.pixels),
                    shape=(
                        self.unwrapped.padding,
                        self.unwrapped.padding * self.env.unwrapped.holder.size,
                    ),
                    dtype=np.uint8,
                ),
                "queue": gym.spaces.Box(
                    low=0,
                    high=len(self.env.unwrapped.pixels),
                    shape=(
                        self.env.unwrapped.padding,
                        self.env.unwrapped.padding * self.env.unwrapped.queue.size,
                    ),
                    dtype=np.uint8,
                ),
            }
        )

        self.legal_actions_mask = np.ones(self.action_space.n)

    def observation(self, observation):
        """Observation wrapper that groups the actions into placements.

        Args:
            observation: The observation to wrap. This is the board without the active tetromino.

        Returns:
            A dictionary containing the grouped board, holder and queue observations.
        """
        board_obs = observation["board"]
        holder_obs = observation["holder"]
        queue_obs = observation["queue"]

        grouped_board_obs = []

        t = self.env.unwrapped.active_tetromino
        for x in range(self.env.unwrapped.width):
            # reset position
            x = self.env.unwrapped.padding + x

            for r in range(4):
                y = 0

                # do rotation
                t = self.env.unwrapped.rotate(t)

                # hard drop
                while not self.env.unwrapped.collision(t, x, y + 1):
                    y += 1

                # append to results
                if not self.env.unwrapped.collision(t, x, y):
                    grouped_board_obs.append(
                        self.env.unwrapped.project_tetromino(t, x, y)
                    )
                else:
                    # this happens when rotation was illegal and the tetromino wasn't dropped at all
                    grouped_board_obs.append(np.ones_like(board_obs))
                    self.legal_actions_mask[
                        (x - self.env.unwrapped.padding) * 4 + r
                    ] = 0

        # concat the results
        grouped_board_obs = np.array(grouped_board_obs)

        return {
            "board": grouped_board_obs.astype(np.uint8),
            "holder": holder_obs.astype(np.uint8),
            "queue": queue_obs.astype(np.uint8),
        }

    def step(self, action):
        """Performs the action.

        Args:
            action: The action to perform.

        Returns:
            The observation, reward, game over, truncated, and info.
        """
        x = action // 4
        r = action % 4

        if self.legal_actions_mask[x * 4 + r] == 0:
            # Do nothing action
            observation, reward, game_over, truncated, info = self.env.step(
                self.env.unwrapped.actions.no_op
            )
            return self.observation(observation), reward, game_over, truncated, info

        new_tetromino = copy.deepcopy(self.env.unwrapped.active_tetromino)

        # Set new x position
        x += self.env.unwrapped.padding
        # Set new rotation
        for _ in range(r):
            new_tetromino = self.env.unwrapped.rotate(new_tetromino)

        # Apply rotation and movement (x,y)
        self.env.unwrapped.x = x
        self.env.unwrapped.active_tetromino = new_tetromino
        observation, reward, game_over, truncated, info = self.env.step(
            self.env.unwrapped.actions.hard_drop
        )
        return self.observation(observation), reward, game_over, truncated, info

    def reset(
        self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None
    ) -> "tuple[dict[str, Any], dict[str, Any]]":
        """Resets the environment.

        Args:
            seed: The seed to use for the random number generator.
            options: The options to use for the environment.

        Returns:
            The observation and info.
        """
        self.legal_actions_mask = np.ones(self.action_space.n)
        observation, info = self.env.reset(seed=seed, options=options)
        return self.observation(observation), info


class GroupedActionsVector(gym.ObservationWrapper):
    """Observation wrapper that groups the actions into placements.

    The action space is the width of the board times 4 (4 rotations).
    """

    def __init__(self, env: Tetris):
        """Initializes the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self.action_space = Discrete((env.unwrapped.width) * 4)
        self.observation_space = Box(
            low=0,
            high=env.unwrapped.height * env.unwrapped.width,
            shape=(
                4 * (
                        # env.unwrapped.width * env.unwrapped.width + # height
                        env.unwrapped.width # max_height
#                         env.unwrapped.width # holes
                ),
            ),
            dtype=np.float32,
        )
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "height": Box(
        #             low=0,
        #             high=env.unwrapped.height,
        #             shape=(
        #                 env.unwrapped.width,
        #                 env.unwrapped.width,1
        #             ),
        #             dtype=np.uint8,
        #         ),
        #         "max_height": Box(
        #             low=0,
        #             high=env.unwrapped.height * env.unwrapped.width,
        #             shape=(
        #                 env.unwrapped.width,1
        #             ),
        #             dtype=np.uint8,
        #         ),
        #         "holes": Box(
        #             low=0,
        #             high=env.unwrapped.height * env.unwrapped.width,
        #             shape=(
        #                 env.unwrapped.width,1
        #             ),
        #             dtype=np.uint8,
        #         ),
        #     }
        # )

        self.legal_actions_mask = np.ones(self.action_space.n)

    # def calculate_height(self, board):
    #     """Calculate the height of each column in the board."""
    #     board = copy.deepcopy(board)
    #     board = board[0:-self.env.unwrapped.padding,self.env.unwrapped.padding:-self.env.unwrapped.padding]
    #     height = np.zeros(board.shape[1], dtype=int)
    #     for col in range(board.shape[1]):
    #         for row in range(board.shape[0]):
    #             if board[row, col] != 0:
    #                 height[col] = board.shape[0] - row
    #                 break
    #     return height

    def calculate_height(self, board):
        """Calculate the height of each column in the board."""
        # Slicing the board to remove padding
        board = board[0:-self.env.unwrapped.padding, self.env.unwrapped.padding:-self.env.unwrapped.padding]

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
        board = board[0:-self.env.unwrapped.padding, self.env.unwrapped.padding:-self.env.unwrapped.padding]

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
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.env.unwrapped.height)
        heights = self.env.unwrapped.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def collision_with_frame(self, tetromino: Tetromino, x: int, y: int) -> bool:
        """Check if the tetromino collides with the board at the given position.

        A collision is detected if the tetromino overlaps with any non-zero cell on the board.
        These non-zero cells represent the padding / bedrock (value 1) or other tetrominoes (values >=2).

        Args:
            tetromino: The tetromino to check for collision.
            x: The x position of the tetromino to check collision for.
            y: The y position of the tetromino to check collision for.

        Returns:
            True if the tetromino collides with the board at the given position, False otherwise.
        """
        # Extract the part of the board that the tetromino would occupy.
        slices = self.env.unwrapped.get_tetromino_slices(tetromino, x, y)
        board_subsection = self.env.unwrapped.board[slices]

        # Check collision using numpy element-wise operations.
        result = np.any(board_subsection[tetromino.matrix > 0] == 1)
        return np.any(board_subsection[tetromino.matrix > 0] == 1)
    def observation(self, observation):
        """Observation wrapper that groups the actions into placements.

        Args:
            observation: The observation to wrap. This is the board without the active tetromino.

        Returns:
            A dictionary containing the grouped board, holder and queue observations.
        """
        board_obs = observation["board"]
        holder_obs = observation["holder"]
        queue_obs = observation["queue"]

        self.legal_actions_mask = np.ones(self.action_space.n)

        grouped_board_obs = []

        t = self.env.unwrapped.active_tetromino
        for x in range(self.env.unwrapped.width):
            # reset position
            x = self.env.unwrapped.padding + x

            for r in range(4):
                y = 0

                # do rotation
                t = self.env.unwrapped.rotate(t)

                # hard drop
                while not self.env.unwrapped.collision(t, x, y + 1):
                    y += 1

                # append to results
                xx = (x - self.env.unwrapped.padding) * 4 + (r+1)%4
                if self.collision_with_frame(t, x, y):
                    self.legal_actions_mask[
                        (x - self.env.unwrapped.padding) * 4 + (r+1)%4
                    ] = 0
                    grouped_board_obs.append(np.ones_like(board_obs))
                elif not self.env.unwrapped.collision(t, x, y):
                    grouped_board_obs.append(
                        self.env.unwrapped.project_tetromino(t, x, y)
                    )
                else:
                    # regular game over
                    grouped_board_obs.append(np.ones_like(board_obs))

        # concat the results
        grouped_board_obs = np.array(grouped_board_obs)

        height_obs = np.array([self.calculate_height(board) for board in grouped_board_obs])
        max_height_obs = np.array([self.calculate_max_height(height) for height in height_obs])
        # holes_obs = np.array([self.calculate_holes(board) for board in grouped_board_obs])
        # # hb_obs = np.array([self.get_bumpiness_and_height(board) for board in grouped_board_obs])
        #
        # # flatten all observations to one array
        # height_obs = height_obs.flatten()
        # max_height_obs = max_height_obs.flatten()
        # holes_obs = holes_obs.flatten()
        # # hb_obs = hb_obs.flatten()

        # obs = np.concatenate((height_obs, max_height_obs, holes_obs)).astype(np.float32)
        obs = max_height_obs.astype(np.float32)
        return obs

    def step(self, action):
        """Performs the action.

        Args:
            action: The action to perform.

        Returns:
            The observation, reward, game over, truncated, and info.
        """
        x = action // 4
        r = action % 4

        if self.legal_actions_mask[action] == 0:
            # observations = np.ones(self.observation_space.shape) * self.observation_space.high
            # reward = self.env.unwrapped.rewards.invalid_action
            # game_over = True
            # truncated = False
            # info = {"action_mask": self.legal_actions_mask}
            # return observations, reward, game_over, truncated, info
            obs, reward, game_over, truncated, info = self.env.unwrapped.step(self.env.unwrapped.actions.no_op)
            reward = self.env.unwrapped.rewards.invalid_action
            info["action_mask"] = self.legal_actions_mask
            return self.observation(obs), reward, game_over, truncated, info

        new_tetromino = copy.deepcopy(self.env.unwrapped.active_tetromino)

        # Set new x position
        x += self.env.unwrapped.padding
        # Set new rotation
        for _ in range(r):
            new_tetromino = self.env.unwrapped.rotate(new_tetromino)

        # Apply rotation and movement (x,y)
        self.env.unwrapped.x = x
        self.env.unwrapped.active_tetromino = new_tetromino
        observation, reward, game_over, truncated, info = self.env.step(
            self.env.unwrapped.actions.hard_drop
        )

        obs = self.observation(observation)
        info["action_mask"] = self.legal_actions_mask
        return obs, reward, game_over, truncated, info

    def reset(
        self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None
    ) -> "tuple[dict[str, Any], dict[str, Any]]":
        """Resets the environment.

        Args:
            seed: The seed to use for the random number generator.
            options: The options to use for the environment.

        Returns:
            The observation and info.
        """
        self.legal_actions_mask = np.ones(self.action_space.n)
        observation, info = self.env.reset(seed=seed, options=options)
        info["action_mask"] = self.legal_actions_mask
        return self.observation(observation), info

    def get_rgb(self, observation):
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
        matrix = self.get_rgb(self.env.unwrapped._get_obs())

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