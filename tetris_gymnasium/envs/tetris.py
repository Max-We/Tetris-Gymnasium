"""Tetris environment for Gymnasium."""
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

ACTIONS = {
    "move_left": 0,
    "move_right": 1,
    "move_down": 2,
    "rotate_clockwise": 3,
    "rotate_counterclockwise": 4,
    "hard_drop": 5,
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
        width=10,
        height=20,
        tetrominoes=STANDARD_TETROMINOES,
        randomizer=BagRandomizer,
    ):
        """Creates a new Tetris environment.

        Args:
            render_mode: The mode to use for rendering. If None, no rendering will be done.
            width: The width of the game board.
            height: The height of the game board.
            tetrominoes: A list of numpy arrays representing the tetrominoes to use.
            randomizer: The randomizer to use for selecting tetrominoes.
        """
        # Dimensions
        self.height: int = height
        self.width: int = width

        # Tetrominoes & Schedule
        self.randomizer: Randomizer = randomizer(len(tetrominoes))
        self.tetrominoes: List[np.ndarray] = tetrominoes
        self.active_tetromino: np.ndarray = self.tetrominoes[
            self.randomizer.get_next_tetromino()
        ]

        # Padding
        self.padding: int = max(max(t.shape) for t in tetrominoes)
        self.width_padded: int = self.width + 2 * self.padding
        self.height_padded: int = self.height + 2 * self.padding

        # Board
        self.board = self.create_board()

        # Position
        self.x: int = 0
        self.y: int = 0

        # Gymnasium
        self.observation_space = Box(
            low=0,
            high=len(self.tetrominoes),
            shape=(self.height, self.width),
            dtype=np.float32,
        )
        self.action_space = Discrete(len(ACTIONS))
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
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        game_over = False
        truncated = False  # Tetris without levels will never truncate
        reward = REWARDS["alife"]

        if action == ACTIONS["move_left"]:
            if not self.collision(self.active_tetromino, self.x - 1, self.y):
                self.x -= 1
        elif action == ACTIONS["move_right"]:
            if not self.collision(self.active_tetromino, self.x + 1, self.y):
                self.x += 1
        elif action == ACTIONS["move_down"]:
            if not self.collision(self.active_tetromino, self.x, self.y + 1):
                self.y += 1
        elif action == ACTIONS["rotate_clockwise"]:
            if not self.collision(
                self.rotate(self.active_tetromino, True), self.x, self.y
            ):
                self.active_tetromino = self.rotate(self.active_tetromino, True)
        elif action == ACTIONS["rotate_counterclockwise"]:
            if not self.collision(
                self.rotate(self.active_tetromino, False), self.x, self.y
            ):
                self.active_tetromino = self.rotate(self.active_tetromino, False)
        elif action == ACTIONS["hard_drop"]:
            # 1. Drop the tetromino and lock it in place
            self.drop_active_tetromino()
            self.place_active_tetromino()
            reward += self.score(self.clear_filled_rows())

            # 2. Spawn the next tetromino and check if the game continues
            game_over = not self.spawn_tetromino()
            if game_over:
                reward = REWARDS["game_over"]

        return self._get_obs(), reward, game_over, truncated, {}

    def reset(
        self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None
    ) -> "tuple[np.ndarray, dict[str, Any]]":
        """Resets the state of the environment.

        Args:
            seed: The random seed to use for the reset.
            options: A dictionary of options to use for the reset.

        Returns: The initial observation of the space.
        """
        super().reset(seed=seed, options=options)

        # Initialize fresh board
        self.board = self.create_board()

        # Reset the randomizer
        self.randomizer.reset(seed=seed)

        # Get the next tetromino and spawn it
        self.active_tetromino = self.tetrominoes[self.randomizer.get_next_tetromino()]
        self.reset_tetromino_position()

        return self._get_obs(), self._get_info()

    def render(self) -> "RenderFrame | list[RenderFrame] | None":
        """Renders the environment in various formats."""
        if self.render_mode == "ansi":
            # Render active tetromino (because it's not on self.board)
            if self.active_tetromino is not None:
                view = self.board.copy()
                slices = self.get_active_tetromino_slices(
                    self.active_tetromino, self.x, self.y
                )
                view[slices] += self.active_tetromino
            else:
                view = self.board

            # Crop padding away as we don't want to render it
            view = self.crop_padding(view)

            # Convert to string
            char_field = np.where(view == 0, ".", view.astype(str))
            field_str = "\n".join("".join(row) for row in char_field)
            return field_str
        elif self.render_mode == "rgb_array":
            # Initialize rgb array
            rgb = np.zeros(
                (self.board.shape[0], self.board.shape[1], 3), dtype=np.uint8
            )
            # Render the board
            rgb[...] = STANDARD_COLORS[self.board]

            # Render active tetromino (because it's not on self.board)
            if self.active_tetromino is not None:
                # Expand to 3 Dimensions for RGB
                active_tetromino_rgb = np.repeat(
                    self.active_tetromino[:, :, np.newaxis], 3, axis=2
                )
                active_tetromino_rgb[...] = STANDARD_COLORS[self.active_tetromino]

                # Apply by masking
                slices = self.get_active_tetromino_slices(
                    self.active_tetromino, self.x, self.y
                )
                rgb[slices] += active_tetromino_rgb

            # Crop padding away as we don't want to render it
            return self.crop_padding(rgb)

        return None

    def spawn_tetromino(self) -> bool:
        """Spawns a new tetromino at the top of the board and checks for collision.

        Returns
            True if the tetromino can be successfully spawned, False otherwise.
        """
        self.active_tetromino = self.tetrominoes[self.randomizer.get_next_tetromino()]
        self.reset_tetromino_position()
        return not self.collision(self.active_tetromino, self.x, self.y)

    def place_active_tetromino(self):
        """Locks the active tetromino in place on the board."""
        slices = self.get_active_tetromino_slices(self.active_tetromino, self.x, self.y)
        self.board[slices] += self.active_tetromino
        self.active_tetromino = None

    def collision(self, tetromino: np.ndarray, x: int, y: int) -> bool:
        """Check if the tetromino collides with the board at the given position.

        A collision is detected if the tetromino overlaps with any non-zero cell on the board.
        These non-zero cells represent the padding / bedrock (value 1) or other tetrominoes (values 2+).

        Args:
            tetromino: The tetromino to check for collision.
            x: The x position of the tetromino to check collision for.
            y: The y position of the tetromino to check collision for.

        Returns:
            True if the tetromino collides with the board at the given position, False otherwise.
        """
        # Extract the part of the board that the tetromino would occupy.
        slices = self.get_active_tetromino_slices(tetromino, x, y)
        board_subsection = self.board[slices]

        # Check collision using numpy element-wise operations.
        return np.any(board_subsection[tetromino > 0] > 0)

    def rotate(self, tetromino: np.ndarray, clockwise=True) -> np.ndarray:
        """Rotate a tetromino by 90 degrees.

        Args:
            tetromino: The tetromino to rotate.
            clockwise: Whether to rotate the tetromino clockwise or counterclockwise.

        Returns:
            The rotated tetromino.
        """
        return np.rot90(tetromino, k=(1 if clockwise else -1))

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

    def get_active_tetromino_slices(
        self, tetromino: np.ndarray, x: int, y: int
    ) -> "tuple(slice, slice)":
        """Get the slices of the active tetromino on the board.

        Returns:
            The slices of the active tetromino on the board.
        """
        tetromino_height, tetromino_width = tetromino.shape
        return tuple((slice(y, y + tetromino_height), slice(x, x + tetromino_width)))

    def reset_tetromino_position(self) -> None:
        """Reset the x and y position of the active tetromino to the center of the board."""
        self.x, self.y = self.width_padded // 2 - self.active_tetromino.shape[0] // 2, 0

    def _get_obs(self) -> np.ndarray:
        """Return the current board as an observation."""
        board_stripped = self.crop_padding(self.board)
        return board_stripped.astype(np.float32)

    def _get_info(self) -> "dict[str, Any]":
        """Return the current game state as info."""
        # Todo: Implement score etc.
        return {}

    def score(self, rows_cleared) -> int:
        """Calculate the score based on the number of lines cleared.

        Args:
            rows_cleared: The number of lines cleared in the last step.

        Returns
            The score for the given number of lines cleared.
        """
        return rows_cleared * REWARDS["clear_line"]

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
