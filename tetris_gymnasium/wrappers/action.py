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

    def __init__(self, env: Tetris, observation_wrappers: "list[gym.ObservationWrapper]" = None):
        """Initializes the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self.action_space = Discrete((env.unwrapped.width) * 4)

        grouped_env_shape = (env.unwrapped.width * 4,)
        single_env_shape = observation_wrappers[-1].observation_space.shape if observation_wrappers else env.observation_space["board"].shape

        self.observation_space = Box(
            low=0,
            high=env.unwrapped.height * env.unwrapped.width,
            shape=(grouped_env_shape + single_env_shape),
            dtype=np.float32,
        )

        self.legal_actions_mask = np.ones(self.action_space.n)
        self.observation_wrappers = observation_wrappers


    def xr_to_action(self, x, r):
        return x * 4 + r

    def action_to_xr(self, action):
        return action // 4, action % 4

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
                if r > 0:
                    t = self.env.unwrapped.rotate(t)

                # hard drop
                while not self.env.unwrapped.collision(t, x, y + 1):
                    y += 1

                # append to results
                if self.collision_with_frame(t, x, y):
                    self.legal_actions_mask[self.xr_to_action(x - self.env.unwrapped.padding,r)] = 0
                    grouped_board_obs.append(np.ones_like(board_obs))
                elif not self.env.unwrapped.collision(t, x, y):
                    grouped_board_obs.append(
                        self.env.unwrapped.project_tetromino(t, x, y)
                    )
                else:
                    # regular game over
                    grouped_board_obs.append(np.ones_like(board_obs))

            t = self.env.unwrapped.rotate(t) # reset rotation (thus far has been rotated 3 times)

        # Apply wrappers
        if self.observation_wrappers is not None:
            for i, observation in enumerate(grouped_board_obs):
                # Recreate the original environment observation
                grouped_board_obs[i] = {
                    "board": observation,
                    "active_tetromino_mask": np.zeros_like(observation),  # Not used in this wrapper
                    "holder": holder_obs,
                    "queue": queue_obs,
                }

                # Validate that observations are equal
                assert grouped_board_obs[i].keys() == self.env.unwrapped.observation_space.keys()

                # Apply wrappers to all the original observations
                for wrapper in self.observation_wrappers:
                    grouped_board_obs[i] = wrapper.observation(grouped_board_obs[i])

        grouped_board_obs = np.array(grouped_board_obs)
        return grouped_board_obs

    def step(self, action):
        """Performs the action.

        Args:
            action: The action to perform.

        Returns:
            The observation, reward, game over, truncated, and info.
        """
        x, r = self.action_to_xr(action)

        if self.legal_actions_mask[action] == 0:
            observations = np.ones(self.observation_space.shape) * self.observation_space.high
            reward = self.env.unwrapped.rewards.invalid_action
            game_over = True
            truncated = False
            info = {"action_mask": self.legal_actions_mask}
            return observations, reward, game_over, truncated, info
            # obs, reward, game_over, truncated, info = self.env.unwrapped.step(
            #     self.env.unwrapped.actions.no_op
            # )
            # reward = self.env.unwrapped.rewards.invalid_action
            # info["action_mask"] = self.legal_actions_mask
            # return self.observation(obs), reward, game_over, truncated, info

        new_tetromino = copy.deepcopy(self.env.unwrapped.active_tetromino)

        # Set new x position
        x += self.env.unwrapped.padding
        # Set new rotation
        for _ in range(r):
            new_tetromino = self.env.unwrapped.rotate(new_tetromino)

        # Apply rotation and movement (x,y)
        self.env.unwrapped.x = x
        self.env.unwrapped.active_tetromino = new_tetromino

        # Perform the action
        observation, reward, game_over, truncated, info = self.env.unwrapped.step(
            self.env.unwrapped.actions.hard_drop
        )

        info["action_mask"] = self.legal_actions_mask
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
        info["action_mask"] = self.legal_actions_mask
        return self.observation(observation), info
