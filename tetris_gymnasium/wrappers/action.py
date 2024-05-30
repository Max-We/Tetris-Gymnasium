"""WIP Action wrapper."""
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

from tetris_gymnasium.envs import Tetris


class GroupedActions(gym.ObservationWrapper):
    def __init__(self, env: Tetris):
        super().__init__(env)
        self.action_space = Discrete(env.unwrapped.width * 4)
        self.observation_space = Box(
            low=0,
            high=len(env.unwrapped.tetrominoes),
            shape=(
                env.unwrapped.width * 4,
                env.unwrapped.height_padded,
                env.unwrapped.width_padded
                + max(env.unwrapped.holder.size, env.unwrapped.queue.size)
                * env.unwrapped.padding
            ),
            dtype=np.uint8,
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
                    grouped_board_obs.append(self.env.unwrapped.project_tetromino(t, x, y))
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

        # Set x position
        self.env.unwrapped.x = x + self.env.unwrapped.padding
        # Set rotation
        for _ in range(r):
            self.env.unwrapped.rotate(self.env.unwrapped.active_tetromino)
        # Drop tetromino
        observation, reward, game_over, truncated, info = self.env.step(self.env.unwrapped.actions.hard_drop)

        return self.observation(observation), reward, game_over, truncated, info
