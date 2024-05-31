"""WIP Action wrapper."""
import copy

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from tetris_gymnasium.envs import Tetris


class GroupedActions(gym.ObservationWrapper):
    def __init__(self, env: Tetris):
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

    def observation(self, observation):
        board_obs = observation["board"]
        holder_obs = observation["holder"]
        queue_obs = observation["queue"]

        grouped_board_obs = []

        t = self.env.unwrapped.active_tetromino
        for x in range(self.env.unwrapped.width):
            # reset position
            x = self.env.unwrapped.padding + x

            for _ in range(4):
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
                    grouped_board_obs.append(np.ones_like(board_obs))

        # concat the results
        grouped_board_obs = np.array(grouped_board_obs)

        return {
            "board": grouped_board_obs.astype(np.uint8),
            "holder": holder_obs.astype(np.uint8),
            "queue": queue_obs.astype(np.uint8),
        }

    def step(self, action):
        x = action // 4
        r = action % 4
        new_tetromino = copy.deepcopy(self.env.unwrapped.active_tetromino)

        # Set new x position
        x += self.env.unwrapped.padding
        # Set new rotation
        for _ in range(r):
            new_tetromino = self.env.unwrapped.rotate(new_tetromino)

        # Check if position is legal
        if self.env.unwrapped.collision(new_tetromino, x, self.env.unwrapped.y):
            # Do nothing action
            observation, reward, game_over, truncated, info = self.env.step(self.env.unwrapped.actions.no_op)
            return self.observation(observation), reward, game_over, truncated, info

        # Apply rotation and movement (x,y)
        self.env.unwrapped.x = x
        self.env.unwrapped.active_tetromino = new_tetromino
        observation, reward, game_over, truncated, info = self.env.step(self.env.unwrapped.actions.hard_drop)
        return self.observation(observation), reward, game_over, truncated, info

    def reset(
        self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None
    ) -> "tuple[dict[str, Any], dict[str, Any]]":
        observation, info = self.env.reset(seed=seed, options=options)
        return self.observation(observation), info
