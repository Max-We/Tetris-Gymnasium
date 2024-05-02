"""Tetris environment for Gymnasium."""
from itertools import permutations
from typing import Any, List

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, RenderFrame
from gymnasium.spaces import Box, Discrete

from tetris_gymnasium.components.randomizer import BagRandomizer, Randomizer
from tetris_gymnasium.util.tetrominoes import STANDARD_COLORS, STANDARD_TETROMINOES

REWARDS = {
    "alife": 0.001,
    "clear_line": 1,
    "game_over": -2,
}


class Tetris(gym.Env):
    """Tetris environment for Gymnasium."""

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 1,
        "render_human": True,
    }

    def __init__(
        self,
        render_mode=None,
        dimensions=(20, 10),
        tetrominoes=STANDARD_TETROMINOES,
        randomizer=BagRandomizer,
    ):
        """Creates a new Tetris environment.

        Args:
            render_mode: The mode to use for rendering. If None, no rendering will be done.
            dimensions: The dimensions of the board as a tuple (height, width).
            tetrominoes: A list of numpy arrays representing the tetrominoes to use.
            randomizer: The randomizer to use for selecting tetrominoes.
        """
        # Dimensions
        self.window_height: int = dimensions[0] * 100
        self.window_width: int = dimensions[1] * 100

        # Tetrominoes & Schedule
        self.randomizer: Randomizer = randomizer(len(tetrominoes))
        self.tetrominoes: List[np.ndarray] = tetrominoes
        self.active_tetromino: np.ndarray = self.tetrominoes[
            self.randomizer.get_next_tetromino()
        ]
        self.dimensions = dimensions
        self.board: np.ndarray = np.zeros(self.dimensions, dtype=np.uint8)

        self.padding = max(max(t.shape) for t in tetrominoes)
        self.pad_mask = (
            np.ones((len(self.dimensions), 2), dtype=np.integer) * self.padding
        )
        self.pad_mask[0, 0] = 0
        self.board = np.pad(
            self.board,
            self.pad_mask,
            mode="constant",
            constant_values=1,
        )

        # Position
        self.pos = np.zeros(len(self.dimensions), dtype=np.uint8)

        all_axes = range(len(self.dimensions))
        self.axis_pairs = list(permutations(all_axes, 2))

        # Game state
        self.game_over: bool = False

        # Gymnasium
        self.observation_space = Box(
            low=0,
            high=len(self.tetrominoes),
            shape=dimensions,
            dtype=np.float32,
        )
        # Actions: Movements + Rotations + Hard Drop
        self.action_space = Discrete(len(dimensions) * 2 + len(self.axis_pairs) + 1)
        self.reward_range = (min(REWARDS.values()), max(REWARDS.values()))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # """
        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        # """
        # self.window = None
        # self.clock = None

    def step(self, action: ActType) -> "tuple[np.ndarray, float, bool, bool, dict]":
        """Perform one step of the environment's dynamics.

        Args:
            action: The action to be executed.

        Returns:
            observation: The observation of the current board as np array.
            reward: Amount of reward returned after previous action.
            done: Whether the episode has ended, in which case further step() calls will return undefined results.
            info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        # Todo: impl. truncation
        # Todo: impl. ActType
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        truncated = False

        if self.active_tetromino is None:
            self.spawn_tetromino()
        else:
            dims = len(self.dimensions)

            # Move: 0 -> dim * 2
            if action in range(dims * 2):
                for i in range(dims * 2):
                    direction = 1 if i % 2 == 0 else -1
                    if i == 1 and direction == -1:
                        # Moving up is illegal
                        break
                    if action == i:
                        axis = i // dims
                        new_pos = self.pos.copy()
                        new_pos[axis] += direction

                        if not self.check_collision(self.active_tetromino, new_pos):
                            self.pos = new_pos
                        break
            elif action in range(dims * 2, dims * 2 + len(self.axis_pairs)):
                # Rotate: dim * 2 -> dim * 4
                for i in range(dims * 2, dims * 2 + len(self.axis_pairs)):
                    clockwise = 1 if i % 2 == 0 else -1
                    if action == i:
                        new_tetromino = np.rot90(
                            self.active_tetromino,
                            k=clockwise,
                            axes=self.axis_pairs[i - (dims * 2)],
                        )
                        if not self.check_collision(new_tetromino, self.pos):
                            self.active_tetromino = new_tetromino
                        break
            else:
                # Hard drop
                self.drop_active_tetromino()
                self.place_active_tetromino()
                return (
                    self._get_obs(),
                    REWARDS["game_over"] * self.clear_filled_rows() + REWARDS["alife"],
                    False,
                    truncated,
                    {},
                )

        if self.game_over:
            return self._get_obs(), REWARDS["game_over"], True, truncated, {}

        return self._get_obs(), REWARDS["alife"], False, truncated, {}

    def reset(
        self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None
    ) -> "tuple[np.ndarray, dict[str, Any]]":
        """Resets the state of the environment.

        Args:
            seed: The random seed to use for the reset.
            options: A dictionary of options to use for the reset.

        Returns: The initial observation of the space.
        """
        # Todo: Implement options
        super().reset(seed=seed, options=options)

        # Initialize fresh board
        self.board: np.ndarray = np.zeros(self.dimensions, dtype=np.uint8)
        self.board = np.pad(
            self.board,
            self.pad_mask,
            mode="constant",
            constant_values=1,
        )

        # Reset the randomizer
        self.randomizer.reset(seed=seed)

        # Get first piece from bag
        self.active_tetromino = self.tetrominoes[self.randomizer.get_next_tetromino()]

        for i in range(len(self.dimensions)):
            self.pos[i] = self.board.shape[i] // 2 - len(self.active_tetromino[i]) // 2
        self.pos[0] = 0

        self.game_over = False

        return self._get_obs(), {}

    def render(self) -> "RenderFrame | list[RenderFrame] | None":
        """Renders the environment as text."""
        if self.render_mode == "ansi":
            # Render active tetromino (because it's not on self.board)
            if self.active_tetromino is not None:
                view = self.board.copy()
                indices = tuple(
                    slice(self.pos[i], self.pos[i] + self.active_tetromino.shape[i])
                    for i in range(self.board.ndim)
                )
                view[indices] += self.active_tetromino
            else:
                view = self.board

            # Crop padding away as we don't want to render it
            stripped_indices = tuple(
                slice(0, -self.padding)
                if i == 0
                else slice(self.padding, -self.padding)
                for i in range(len(self.dimensions))
            )
            view = view[stripped_indices]

            # Convert to string
            char_field = np.where(view == 0, ".", view.astype(str))
            field_str = "\n".join("".join(row) for row in char_field)
            return field_str
        elif self.render_mode == "rgb_array":
            # Initialize rgb array
            rgb = np.zeros(self.board.shape + (3,), dtype=np.uint8)
            rgb[...] = STANDARD_COLORS[self.board]

            # Render active tetromino (because it's not on self.board)
            if self.active_tetromino is not None:
                # Render active tetromino (because it's not on self.board)
                # Assuming active_tetromino is a smaller n-dimensional array within board's dimensions
                active_dims = self.active_tetromino.ndim
                indices = tuple(
                    slice(self.pos[i], self.pos[i] + self.active_tetromino.shape[i])
                    for i in range(active_dims)
                )

                # Ensure RGB channels for drawing
                color_channels = 3
                expanded_dims = self.active_tetromino.shape + (color_channels,)
                active_tetromino_rgb = np.zeros(expanded_dims, dtype=np.uint8)
                active_tetromino_rgb[..., :] = STANDARD_COLORS[self.active_tetromino]

                # Place expanded and colored tetromino into the board (which is assumed to have space for colors)
                rgb[indices] += active_tetromino_rgb

            # Crop padding away as we don't want to render it
            stripped_indices = tuple(
                slice(0, -self.padding)
                if i == 0
                else slice(self.padding, -self.padding)
                for i in range(active_dims)
            )
            rgb = rgb[stripped_indices]
            return rgb

        return None

    def spawn_tetromino(self):
        """Spawns a new tetromino at the top of the board.

        If the tetromino collides with the top of the board, the game is over.
        """
        self.active_tetromino = self.tetrominoes[self.randomizer.get_next_tetromino()]
        for i in range(len(self.pos)):
            self.pos[i] = self.board.shape[i] // 2 - len(self.active_tetromino[i]) // 2
        self.pos[0] = 0
        self.game_over = self.check_collision(self.active_tetromino, self.pos)

    def place_active_tetromino(self):
        """Locks the active tetromino in place on the board."""
        # Boundary checks
        indices = tuple(
            slice(self.pos[i], self.pos[i] + self.active_tetromino.shape[i])
            for i in range(self.board.ndim)
        )
        self.board[indices] += self.active_tetromino
        self.active_tetromino = None

    def check_collision(self, tetromino: np.ndarray, pos: np.ndarray) -> bool:
        """Check if the tetromino collides with the board at the given position.

        The collision check is performed in two steps:
        1. Boundary check: Check if the tetromino is within the boundaries of the board.
        2. Overlap check: Check if the tetromino overlaps with any non-zero cells on the board.

        The overlap check is implemented using numpy element-wise operations, which are
        more efficient than using loops to check each cell individually.

        Args:
            tetromino: The tetromino to check for collision.
            x: The x position of the tetromino to check collision for.
            y: The y position of the tetromino to check collision for.

        Returns:
            True if the tetromino collides with the board at the given position, False otherwise.
        """
        # w = pos
        # e = tetromino.shape
        # r = self.dimensions

        indices = tuple(
            slice(pos[i], pos[i] + tetromino.shape[i])
            for i in range(len(self.dimensions))
        )
        board_subarray = self.board[indices]

        # Render active tetromino (because it's not on self.board)
        # if tetromino is not None:
        #     view = self.board.copy()
        #     indices = tuple(
        #         slice(pos[i], pos[i] + tetromino.shape[i]) for i in range(len(self.board.shape)))
        #     view[indices] += tetromino

        # Check collision using numpy element-wise operations.
        # This checks if any corresponding cells (both non-zero) of the subarray and the
        # tetromino overlap, indicating a collision.
        return np.any(board_subarray[tetromino > 0] > 0)

    def rotate(self, tetromino: np.ndarray, clockwise=True) -> np.ndarray:
        """Rotate the given tetromino by 90 degrees.

        Args:
            tetromino: The tetromino to rotate.
            clockwise: Whether to rotate the tetromino clockwise or counterclockwise.

        Returns:
            The rotated tetromino.
        """
        return np.rot90(tetromino, k=(1 if clockwise else -1))

    def drop_active_tetromino(self):
        """Drop the active tetromino to the lowest possible position on the board."""
        # Todo: is there a more efficient way to hard drop without loop?
        new_pos = self.pos.copy()
        while not self.check_collision(self.active_tetromino, new_pos):
            new_pos[0] += 1

        new_pos[0] -= 1
        self.pos = new_pos

    def clear_filled_rows(self) -> int:
        """Clear any filled rows on the board.

        The clearing is performed using numpy by indexing only the rows that are not filled and
        concatenating them with a new top part of the board that contains zeros.

        With this implementation, the clearing operation is efficient and does not require looping.

        Returns:
            The number of rows that were cleared.
        """
        # Check for filled rows. A row is filled if it does not contain any zeros.
        filled_rows = np.all(
            [
                (~(self.board == 0).any(axis=ax)) & (~(self.board == 1).all(axis=ax))
                for ax in range(1, self.board.ndim)
            ],
            axis=0,
        )
        num_filled = np.sum(filled_rows)

        if num_filled > 0:
            # Identify the rows that are not filled.
            unfilled_rows = self.board[~filled_rows]

            # Create a new top part of the board with zeros to compensate for the cleared rows.
            free_space = np.zeros((num_filled,) + self.dimensions[1:], dtype=np.uint8)
            mm = self.pad_mask.copy()
            mm[0] *= 0
            free_space = np.pad(
                free_space,
                mm,
                mode="constant",
                constant_values=1,
            )

            # Concatenate the new top with the unfilled rows to form the updated board.
            self.board[:] = np.concatenate((free_space, unfilled_rows), axis=0)

        return num_filled

    def _get_obs(self) -> np.ndarray:
        """Return the current board as an observation."""
        stripped_indices = tuple(
            slice(0, -self.padding) if i == 0 else slice(self.padding, -self.padding)
            for i in range(len(self.dimensions))
        )
        board_stripped = self.board[stripped_indices]
        return board_stripped.astype(np.float32)

    #     super().close()
    #
    # @property
    # def unwrapped(self) -> Env[ObsType, ActType]:
    #     return super().unwrapped
    #
    # @property
    # def np_random(self) -> np.random.Generator:
    #     return super().np_random
    #
    # def __str__(self):
    #     return super().__str__()
    #
    # def __enter__(self):
    #     return super().__enter__()
    #
    # def __exit__(self, *args: Any):
    #     return super().__exit__(*args)
    #
    # def get_wrapper_attr(self, name: str) -> Any:
    #     return super().get_wrapper_attr(name)
