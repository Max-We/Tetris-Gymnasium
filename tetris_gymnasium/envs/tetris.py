"""Tetris environment for Gymnasium."""
from copy import copy
from dataclasses import fields
from typing import Any, List

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, RenderFrame
from gymnasium.spaces import Box, Discrete

from tetris_gymnasium.components.tetromino import Pixel, Tetromino
from tetris_gymnasium.components.tetromino_holder import TetrominoHolder
from tetris_gymnasium.components.tetromino_queue import TetrominoQueue
from tetris_gymnasium.components.tetromino_randomizer import BagRandomizer, Randomizer
from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_gymnasium.mappings.rewards import RewardsMapping


class Tetris(gym.Env):
    """Tetris environment for Gymnasium."""

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 1,
        "render_human": True,
    }

    BASE_PIXELS = [Pixel(0, [0, 0, 0]), Pixel(1, [128, 128, 128])]  # Empty  # Bedrock

    TETROMINOES = [
        Tetromino(
            0,
            [0, 240, 240],
            np.array(
                [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
            ),
        ),  # I
        Tetromino(1, [240, 240, 0], np.array([[1, 1], [1, 1]], dtype=np.uint8)),  # O
        Tetromino(
            2,
            [160, 0, 240],
            np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),
        ),  # T
        Tetromino(
            3, [0, 240, 0], np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
        ),  # S
        Tetromino(
            4, [240, 0, 0], np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
        ),  # Z
        Tetromino(
            5, [0, 0, 240], np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        ),  # J
        Tetromino(
            6,
            [240, 160, 0],
            np.array([[0, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=np.uint8),
        ),  # L
    ]

    def __init__(
        self,
        render_mode=None,
        width=10,
        height=20,
        randomizer: Randomizer = BagRandomizer,
        holder: TetrominoHolder = TetrominoHolder,
        queue: TetrominoQueue = TetrominoQueue,
        actions_mapping=ActionsMapping(),
        rewards_mapping=RewardsMapping(),
        base_pixels=None,
        tetrominoes=None,
    ):
        """Creates a new Tetris environment.

        Args:
            render_mode: The mode to use for rendering. If None, no rendering will be done.
            width: The width of the board.
            height: The height of the board.
            randomizer: The :class:`Randomizer` to use for selecting tetrominoes.
            holder: The :class:`TetrominoHolder` to use for storing tetrominoes.
            queue: The :class:`TetrominoQueue` to use for holding tetrominoes temporarily.
            actions_mapping: The mapping for the actions that the agent can take.
            rewards_mapping: The mapping for the rewards that the agent can receive.
            base_pixels: A list of base (non-Tetromino) :class:`Pixel` to use for the environment (e.g. empty, bedrock).
            tetrominoes: A list of :class:`Tetromino` to use in the environment.
        """
        # Dimensions
        self.height: int = height
        self.width: int = width

        # Base Pixels
        if base_pixels is None:
            self.base_pixels = self.BASE_PIXELS

        # Tetrominoes
        if tetrominoes is None:
            tetrominoes = self.TETROMINOES
        self.tetrominoes: List[Tetromino] = self.offset_tetromino_id(
            tetrominoes, len(self.base_pixels)
        )
        self.active_tetromino: Tetromino = None

        # Pixels
        self.pixels: List[Pixel] = self.parse_pixels(self.tetrominoes)

        # Padding
        self.padding: int = max(max(t.matrix.shape) for t in self.tetrominoes)
        self.width_padded: int = self.width + 2 * self.padding
        self.height_padded: int = self.height + self.padding

        # Board
        self.board = self.create_board()

        # Utilities
        self.queue = queue(randomizer(len(tetrominoes)), 5)
        self.holder = holder()
        self.has_swapped = False

        # Position
        self.x: int = 0
        self.y: int = 0

        # Gymnasium
        self.observation_space = gym.spaces.Dict(
            {
                "board": Box(
                    low=0,
                    high=len(self.pixels),
                    shape=(self.height_padded, self.width_padded),
                    dtype=np.float32,
                ),
                "holder": Box(
                    low=0,
                    high=len(self.pixels),
                    shape=(
                        self.padding,
                        self.padding * self.holder.size,
                    ),
                    dtype=np.float32,
                ),
                "queue": gym.spaces.Box(
                    low=0,
                    high=len(self.pixels),
                    shape=(
                        self.padding,
                        self.padding * self.queue.size,
                    ),
                    dtype=np.float32,
                ),
            }
        )

        # Mappings for rewards  & actions (readability in code)
        self.actions = actions_mapping
        self.rewards = rewards_mapping

        self.action_space = Discrete(len(fields(self.actions)))
        self.reward_range = (
            min(vars(self.rewards).values()),
            max(vars(self.rewards).values()),
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_name = None

    def step(self, action: ActType) -> "tuple[dict, float, bool, bool, dict]":
        """Perform one step of the environment's dynamics.

        Args:
            action: The action to be executed.

        Returns:
            observation: The observation of the current board as np array.
            reward: Amount of reward returned after previous action.
            done: Whether the episode has ended, in which case further step() calls will return undefined results.
            info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        game_over = False
        truncated = False  # Tetris without levels will never truncate
        reward = self.rewards.alife

        if action == self.actions.move_left:
            if not self.collision(self.active_tetromino, self.x - 1, self.y):
                self.x -= 1
        elif action == self.actions.move_right:
            if not self.collision(self.active_tetromino, self.x + 1, self.y):
                self.x += 1
        elif action == self.actions.move_down:
            if not self.collision(self.active_tetromino, self.x, self.y + 1):
                self.y += 1
        elif action == self.actions.rotate_clockwise:
            if not self.collision(
                self.rotate(self.active_tetromino, True), self.x, self.y
            ):
                self.active_tetromino = self.rotate(self.active_tetromino, True)
        elif action == self.actions.rotate_counterclockwise:
            if not self.collision(
                self.rotate(self.active_tetromino, False), self.x, self.y
            ):
                self.active_tetromino = self.rotate(self.active_tetromino, False)
        elif action == self.actions.swap:
            if not self.has_swapped:
                # Swap the active tetromino with the one in the holder (saves orientation)
                self.active_tetromino = self.holder.swap(self.active_tetromino)
                self.has_swapped = True
                if self.active_tetromino is None:
                    # If the holder is empty, spawn the next tetromino
                    # No need for collision check, as the holder is only empty at the start
                    self.spawn_tetromino()
                else:
                    self.reset_tetromino_position()
        elif action == self.actions.hard_drop:
            # 1. Drop the tetromino and lock it in place
            self.drop_active_tetromino()
            self.place_active_tetromino()
            reward += self.score(self.clear_filled_rows())

            # 2. Spawn the next tetromino and check if the game continues
            game_over = not self.spawn_tetromino()
            if game_over:
                reward = self.rewards.game_over

            # 3. Reset the swap flag (agent can swap once per tetromino)
            self.has_swapped = False

        # Cheap fake gravity
        if not self.collision(self.active_tetromino, self.x, self.y + 1):
            self.y += 1
        else:
            # 1. Drop the tetromino and lock it in place
            self.drop_active_tetromino()
            self.place_active_tetromino()
            reward += self.score(self.clear_filled_rows())

            # 2. Spawn the next tetromino and check if the game continues
            game_over = not self.spawn_tetromino()
            if game_over:
                reward = self.rewards.game_over

            # 3. Reset the swap flag (agent can swap once per tetromino)
            self.has_swapped = False

        return self._get_obs(), reward, game_over, truncated, self._get_info()

    def reset(
        self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None
    ) -> "tuple[dict[str, Any], dict[str, Any]]":
        """Resets the state of the environment.

        As with all Gymnasium environments, the reset method is called once at the beginning of an episode.

        Args:
            seed: The random seed to use for the reset.
            options: A dictionary of options to use for the reset.

        Returns: The initial observation of the space.
        """
        super().reset(seed=seed, options=options)

        # Initialize fresh board
        self.board = self.create_board()

        # Reset the randomizer
        self.queue.reset(seed=seed)

        # Get the next tetromino and spawn it
        self.active_tetromino = self.tetrominoes[self.queue.get_next_tetromino()]
        self.reset_tetromino_position()

        # Holder
        self.holder.reset()
        self.has_swapped = False

        # Render
        self.window_name = None

        return self._get_obs(), self._get_info()

    def render(self) -> "RenderFrame | list[RenderFrame] | None":
        """Renders the environment in various formats."""
        if self.render_mode == "ansi":
            # Render active tetromino (because it's not on self.board)
            projection = self.project_active_tetromino()

            # Crop padding away as we don't want to render it
            projection = self.crop_padding(projection)

            # Convert to string
            char_field = np.where(projection == 0, ".", projection.astype(str))
            field_str = "\n".join("".join(row) for row in char_field)
            return field_str
        elif self.render_mode == "human" or self.render_mode == "rgb_array":
            # Initialize rgb array
            rgb = np.zeros(
                (self.board.shape[0], self.board.shape[1], 3), dtype=np.uint8
            )
            # Render the board
            colors = np.array(list(p.color_rgb for p in self.pixels), dtype=np.uint8)
            rgb[...] = colors[self.board]

            # Render active tetromino (because it's not on self.board)
            if self.active_tetromino is not None:
                # Expand to 3 Dimensions for RGB
                active_tetromino_rgb = np.repeat(
                    self.active_tetromino.matrix[:, :, np.newaxis], 3, axis=2
                )
                active_tetromino_rgb[...] = colors[self.active_tetromino.matrix]

                # Apply by masking
                slices = self.get_tetromino_slices(
                    self.active_tetromino, self.x, self.y
                )
                rgb[slices] += active_tetromino_rgb

            # Crop padding away as we don't want to render it
            rgb = self.crop_padding(rgb)

            if self.render_mode == "rgb_array":
                return rgb

            if self.render_mode == "human":
                if self.window_name is None:
                    self.window_name = "Tetris Gymnasium"
                    cv2.namedWindow(self.window_name, cv2.WINDOW_GUI_NORMAL)
                    cv2.resizeWindow(self.window_name, 200, 400)
                cv2.imshow(self.window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        return None

    def spawn_tetromino(self) -> bool:
        """Spawns a new tetromino at the top of the board and checks for collision.

        Returns
            True if the tetromino can be successfully spawned, False otherwise.
        """
        self.active_tetromino = self.tetrominoes[self.queue.get_next_tetromino()]
        self.reset_tetromino_position()
        return not self.collision(self.active_tetromino, self.x, self.y)

    def place_active_tetromino(self):
        """Locks the active tetromino in place on the board."""
        self.board = self.project_active_tetromino()
        self.active_tetromino = None

    def collision(self, tetromino: Tetromino, x: int, y: int) -> bool:
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
        slices = self.get_tetromino_slices(tetromino, x, y)
        board_subsection = self.board[slices]

        # Check collision using numpy element-wise operations.
        return np.any(board_subsection[tetromino.matrix > 0] > 0)

    def rotate(self, tetromino: Tetromino, clockwise=True) -> Tetromino:
        """Rotate a tetromino by 90 degrees.

        Args:
            tetromino: The tetromino to rotate.
            clockwise: Whether to rotate the tetromino clockwise or counterclockwise.

        Returns:
            The rotated tetromino.
        """
        return Tetromino(
            tetromino.id,
            tetromino.color_rgb,
            np.rot90(tetromino.matrix, k=(1 if clockwise else -1)),
        )

    def drop_active_tetromino(self):
        """Drop the active tetromino to the lowest possible position on the board."""
        while not self.collision(self.active_tetromino, self.x, self.y + 1):
            self.y += 1

    def clear_filled_rows(self) -> int:
        """Clear any filled rows on the board.

        The clearing is performed using numpy by indexing only the rows that are not filled and
        concatenating them with a new top part of the board that contains zeros.

        With this implementation, the clearing operation is efficient and does not require loops.

        Returns:
            The number of rows that were cleared.
        """
        # A row is filled if it doesn't contain any free space (0) and doesn't contain any bedrock / padding (1).
        filled_rows = (~(self.board == 0).any(axis=1)) & (
            ~(self.board == 1).all(axis=1)
        )
        n_filled = np.sum(filled_rows)

        if n_filled > 0:
            # Identify the rows that are not filled.
            unfilled_rows = self.board[~filled_rows]

            # Create a new top part of the board with free space (0) to compensate for the cleared rows.
            free_space = np.zeros((n_filled, self.width), dtype=np.uint8)
            free_space = np.pad(
                free_space,
                ((0, 0), (self.padding, self.padding)),
                mode="constant",
                constant_values=1,
            )

            # Concatenate the new top with the unfilled rows to form the updated board.
            self.board[:] = np.concatenate((free_space, unfilled_rows), axis=0)

        return n_filled

    def crop_padding(self, matrix: np.ndarray) -> np.ndarray:
        """Crop the padding from the given matrix.

        The Tetris board has padding on all sides except the top to simplify collision detection.
        This method crops the padding from the given matrix to return the actual board, which is useful for rendering.

        Returns
            The matrix with the padding cropped.
        """
        return matrix[0 : -self.padding, self.padding : -self.padding]

    def get_tetromino_slices(
        self, tetromino: Tetromino, x: int, y: int
    ) -> "tuple(slice, slice)":
        """Get the slices of the active tetromino on the board.

        Returns:
            The slices of the active tetromino on the board.
        """
        tetromino_height, tetromino_width = tetromino.matrix.shape
        return tuple((slice(y, y + tetromino_height), slice(x, x + tetromino_width)))

    def reset_tetromino_position(self) -> None:
        """Reset the x and y position of the active tetromino to the center of the board."""
        self.x, self.y = (
            self.width_padded // 2 - self.active_tetromino.matrix.shape[0] // 2,
            0,
        )

    def project_active_tetromino(self):
        """Project the active tetromino on the board.

        By default, the active (moving) tetromino is not part of the board. This function projects the active tetromino
        on the board to render it.
        """
        projection = self.board.copy()
        slices = self.get_tetromino_slices(self.active_tetromino, self.x, self.y)
        projection[slices] += self.active_tetromino.matrix
        return projection

    def _get_obs(self) -> "dict[str, Any]":
        """Return the current board as an observation."""
        # Include the active tetromino on the board for the observation.
        board_obs = self.project_active_tetromino()

        max_size = self.padding

        # Holder
        holder_tetrominoes = self.holder.get_tetrominoes()
        if len(holder_tetrominoes) > 0:
            # Pad all tetrominoes to be the same size
            for index, t in enumerate(holder_tetrominoes):
                holder_tetrominoes[index] = np.pad(
                    t.matrix,
                    (
                        (0, max_size - t.matrix.shape[0]),
                        (0, max_size - t.matrix.shape[1]),
                    ),
                )
            # Concatenate all tetrominoes horizontally
            holder_obs = np.hstack(holder_tetrominoes)
        else:
            holder_obs = np.ones((max_size, max_size * self.holder.size))

        # Queue
        queue_tetrominoes = self.queue.get_queue()
        for index, t_id in enumerate(queue_tetrominoes):
            # Pad all tetrominoes to be the same size
            t = copy(self.tetrominoes[t_id])
            t.matrix = np.pad(
                t.matrix,
                ((0, max_size - t.matrix.shape[0]), (0, max_size - t.matrix.shape[1])),
            )
            # Safe padded result back to the array
            queue_tetrominoes[index] = t.matrix
        # Concatenate all tetrominoes horizontally
        queue_obs = np.hstack(queue_tetrominoes)

        return {
            "board": board_obs.astype(np.float32),
            "holder": holder_obs.astype(np.float32),
            "queue": queue_obs.astype(np.float32),
        }

    def _get_info(self) -> dict:
        """Return the current game state as info."""
        return {}

    def score(self, rows_cleared) -> int:
        """Calculate the score based on the number of lines cleared.

        Args:
            rows_cleared: The number of lines cleared in the last step.

        Returns
            The score for the given number of lines cleared.
        """
        return rows_cleared * self.rewards.clear_line

    def create_board(self) -> np.ndarray:
        """Create a new board with the given dimensions."""
        board = np.zeros((self.height, self.width), dtype=np.uint8)
        board = np.pad(
            board,
            ((0, self.padding), (self.padding, self.padding)),
            mode="constant",
            constant_values=1,
        )
        return board

    def parse_pixels(self, tetrominoes: "List[Tetromino]") -> "List[Pixel]":
        """Creates a list of pixels from the base pixels and the tetrominoes.

        Pixels are used to represent the board and the tetrominoes in the environment.

        Args:
            tetrominoes: The tetrominoes to add to the base pixels.

        Returns:
            The list of pixels for the environment.
        """
        return self.base_pixels + [
            Pixel(t.id + len(self.base_pixels), t.color_rgb) for t in tetrominoes
        ]

    def offset_tetromino_id(
        self, tetrominoes: "List[Tetromino]", offset: int
    ) -> "List[Tetromino]":
        """In order to make the tetrominos distinguishable, each tetromino should have a unique value.

        The tetrominoes already possess a unique ID, but the matrix should also be updated to reflect this.
        Additionally, the tetrominoes should be offset by a certain value to avoid conflicts with the board.
        The board already contains a number of pixels which are not part of the tetrominoes (empty cells, bedrock).
        So, the tetrominoes should be offset by the number of pixels in the board that are  not tetrominoes.

        Args:
            tetrominoes: The tetrominoes to preprocess.
            offset: The offset to apply to the tetrominoes. This is usually the number of non-tetromino pixels in the board.

        Returns:
            The preprocessed tetrominoes (= id and matrix values offset by number of non-tetromino pixels).
        """
        for i in range(len(tetrominoes)):
            tetrominoes[i].id += offset
            tetrominoes[i].matrix = tetrominoes[i].matrix * (i + offset)

        return tetrominoes
