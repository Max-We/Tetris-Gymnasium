"""Wrapper that changes the observation and actions space into grouped actions.

This wrapper introduces action-grouping as commonly used in current Tetris RL approaches. An example of this idea
can be found in "Playing Tetris with Deep Reinforcement Learning (Stevens & Pradhan)."
"""
import copy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from tetris_gymnasium.components.tetromino import Tetromino
from tetris_gymnasium.envs import Tetris


class GroupedActionsObservations(gym.ObservationWrapper):
    """Wrapper that changes the observation and actions space into grouped actions.

    This wrapper introduces action-grouping as commonly used in current Tetris RL approaches. An example of this idea
    can be found in "*Playing Tetris with Deep Reinforcement Learning* (Stevens & Pradhan)."

    **Action space**
        For each column on the board, the agent can choose between four different rotations. This results in a total of
        `width * 4` possible actions. Therefore, the action space is a `Discrete` space with `width * 4` possible actions.
        The value is interpreted as column index and number of rotations in ascending order. So the actions [0, 1, 2, 3]
        correspond to the first column and the tetromino rotated 0, 1, 2, 3 times respectively. The actions [4, 5, 6, 7]
        correspond to the second column and the tetromino rotated 0, 1, 2, 3 times respectively, and so on.

    **Observation space**
        For each possible action, the wrapper generates a new observation. This means, that an additional dimension of size
        `width * 4` is added to the observation space. Observation wrappers have to be passed to the constructor to apply
        them to the generated observations instead of wrapping them around the `GroupedActions` wrapper.

    **Legal actions**
        Because the action space is static but the game state is dynamic, some actions might be illegal. For this reason,
        the wrapper generates a mask that indicates which actions are legal. This mask is stored in the `legal_actions_mask`
        attribute. If an illegal action is taken, the wrapper can either terminate the episode or return a penalty reward.
        The action mask is returned in the info dictionary under the key `action_mask`. Note that only actions which would
        result in a collision with the frame are considered illegal. Actions which would result in a game over (stack too
        high) are not considered illegal.
    """

    def __init__(
        self,
        env: Tetris,
        observation_wrappers: "list[gym.ObservationWrapper]" = None,
        terminate_on_illegal_action: bool = True,
    ):
        """Initializes the GroupedActions wrapper.

        Args:
            env: The environment to wrap.
            observation_wrappers: The observation wrappers to apply to the individual observation.
            terminate_on_illegal_action: Whether to terminate the episode if an illegal action is taken.
        """
        super().__init__(env)
        self.action_space = Discrete((env.unwrapped.width) * 4)

        grouped_env_shape = (env.unwrapped.width * 4,)
        single_env_shape = (
            observation_wrappers[-1].observation_space.shape
            if observation_wrappers
            else env.observation_space["board"].shape
        )

        self.observation_space = Box(
            low=0,
            high=env.unwrapped.height * env.unwrapped.width,
            shape=(grouped_env_shape + single_env_shape),
            dtype=np.float32,
        )

        self.legal_actions_mask = np.ones(self.action_space.n)
        self.observation_wrappers = observation_wrappers
        self.terminate_on_illegal_action = terminate_on_illegal_action

    def encode_action(self, x, r):
        """Convert x-position and number of rotations `r` to action id.

        Args:
            x: The x position.
            r: The number of rotations.

        Returns:
            The action id.
        """
        return x * 4 + r

    def decode_action(self, action):
        """Converts the action id to the x-position and number of rotations `r`.

        Args:
            action: The action id to convert.

        Returns:
            The x-position and number of rotations.
        """
        return action // 4, action % 4

    def collision_with_frame(self, tetromino: Tetromino, x: int, y: int) -> bool:
        """Check if the tetromino collides with the frame.

        Only collisions with the frame are checked, not with other tetrominos on the board. This is used to detect
        illegal actions. By this definition, actions that end the game (e.g. stack tetromino above the frame) are
        considered legal. If this wasn't the case, the agent could "cheat" by recognizing that the game would be lost
        before taking an action.

        Args:
            tetromino: The tetromino to check for collision.
            x: The x position of the tetromino.
            y: The y position of the tetromino.

        Returns:
            True if the tetromino collides with the frame, False otherwise.
        """
        # Extract the part of the board that the tetromino would occupy.
        slices = self.env.unwrapped.get_tetromino_slices(tetromino, x, y)
        board_subsection = self.env.unwrapped.board[slices]

        # Check collision using numpy element-wise operations.
        return np.any(board_subsection[tetromino.matrix > 0] == 1)

    def observation(self, observation):
        """Observation wrapper that groups the actions into placements and applies additional wrappers (optional).

        This function also generates the legal-action mask.

        Args:
            observation: The original observation from the base environment.

        Returns:
            The grouped observation.
        """
        board_obs = observation["board"]
        holder_obs = observation["holder"]
        queue_obs = observation["queue"]

        self.legal_actions_mask = np.ones(self.action_space.n)

        grouped_board_obs = []

        if self.env.unwrapped.game_over:
            # game over (previous step)
            np.zeros(self.observation_space.shape)

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

                # # append to results
                # if self.collision_with_frame(t, x, y):
                #     self.legal_actions_mask[
                #         self.encode_action(x - self.env.unwrapped.padding, r)
                #     ] = 0
                #     grouped_board_obs.append(np.ones_like(board_obs))
                # elif not self.env.unwrapped.collision(t, x, y):
                #     grouped_board_obs.append(
                #         self.env.unwrapped.project_tetromino(t, x, y)
                #     )
                # else:
                #     # regular game over
                #     grouped_board_obs.append(np.zeros_like(board_obs))

                # append to results

                if self.collision_with_frame(t, x, y):
                    # illegal action
                    self.legal_actions_mask[
                        self.encode_action(x - self.env.unwrapped.padding, r)
                    ] = 0
                    grouped_board_obs.append(np.ones_like(board_obs))
                elif self.env.unwrapped.collision(t, x, y):
                    # game over placement
                    grouped_board_obs.append(np.zeros_like(board_obs))
                else:
                    # regular placement
                    grouped_board_obs.append(
                        self.env.unwrapped.clear_filled_rows(
                            self.env.unwrapped.project_tetromino(t, x, y)
                        )[0]
                    )

            t = self.env.unwrapped.rotate(
                t
            )  # reset rotation (thus far has been rotated 3 times)

        # Apply wrappers
        if self.observation_wrappers is not None:
            for i, observation in enumerate(grouped_board_obs):
                # Recreate the original environment observation
                grouped_board_obs[i] = {
                    "board": observation,
                    "active_tetromino_mask": np.zeros_like(
                        observation
                    ),  # Not used in this wrapper
                    "holder": holder_obs,
                    "queue": queue_obs,
                }

                # Validate that observations are equal
                assert (
                    grouped_board_obs[i].keys()
                    == self.env.unwrapped.observation_space.keys()
                )

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
        x, r = self.decode_action(action)

        if self.legal_actions_mask[action] == 0:
            if self.terminate_on_illegal_action:
                observation = (
                    np.ones(self.observation_space.shape) * self.observation_space.high
                )
                game_over, truncated = True, False
                info = {"action_mask": self.legal_actions_mask, "lines_cleared": 0}
            else:
                (
                    observation,
                    reward,
                    game_over,
                    truncated,
                    info,
                ) = self.env.unwrapped.step(self.env.unwrapped.actions.no_op)
                observation = self.observation(observation)
                info["action_mask"] = self.legal_actions_mask

            reward = self.env.unwrapped.rewards.invalid_action
            return observation, reward, game_over, truncated, info

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
        board = observation
        if self.observation_wrappers:
            for wrapper in self.observation_wrappers:
                board = wrapper.observation(board)
        info["board"] = board

        observation = self.observation(observation)  # generates legal_action_mask
        info["action_mask"] = self.legal_actions_mask

        return observation, reward, game_over, truncated, info

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
        board = observation
        if self.observation_wrappers:
            for wrapper in self.observation_wrappers:
                board = wrapper.observation(board)
        info["board"] = board

        observation = self.observation(observation)  # generates legal_action_mask
        info["action_mask"] = self.legal_actions_mask

        return observation, info
