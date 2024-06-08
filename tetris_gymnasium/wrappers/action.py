"""Observation wrapper that groups the actions into placements.

The action space is the width of the board times 4 (4 rotations).
"""
import copy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

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
