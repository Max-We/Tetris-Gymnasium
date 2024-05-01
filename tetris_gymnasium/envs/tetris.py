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
        self.window_width: int = width * 100
        self.window_height: int = height * 100

        # Tetrominoes & Schedule
        self.randomizer: Randomizer = randomizer(len(tetrominoes))
        self.tetrominoes: List[np.ndarray] = tetrominoes
        self.active_tetromino: np.ndarray = self.tetrominoes[
            self.randomizer.get_next_tetromino()
        ]
        self.board: np.ndarray = np.zeros((self.height, self.width), dtype=np.uint8)

        # Position
        self.x: int = 0
        self.y: int = 0

        # Game state
        self.game_over: bool = False

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
        # Todo: impl. truncation
        # Todo: impl. ActType
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        truncated = False

        if self.active_tetromino is None:
            self.spawn_tetromino()
        else:
            if action == ACTIONS["move_left"]:  # move left
                if not self.check_collision(self.active_tetromino, self.x - 1, self.y):
                    self.x -= 1
            elif action == ACTIONS["move_right"]:  # move right
                if not self.check_collision(self.active_tetromino, self.x + 1, self.y):
                    self.x += 1
            elif action == ACTIONS["rotate_clockwise"]:  # rotate clockwise
                if not self.check_collision(
                    self.rotate(self.active_tetromino, True), self.x, self.y
                ):
                    self.active_tetromino = self.rotate(self.active_tetromino, True)
            elif (
                action == ACTIONS["rotate_counterclockwise"]
            ):  # rotate counterclockwise
                if not self.check_collision(
                    self.rotate(self.active_tetromino, False), self.x, self.y
                ):
                    self.active_tetromino = self.rotate(self.active_tetromino, False)
            elif action == ACTIONS["hard_drop"]:  # hard drop
                self.drop_active_tetromino()
                self.place_active_tetromino()
                return (
                    self._get_obs(),
                    REWARDS["game_over"] * self.clear_filled_rows() + REWARDS["alife"],
                    False,
                    truncated,
                    {},
                )
            elif action == ACTIONS["move_down"]:  # move down (gravity)
                # Todo: Separate move_down from gravity
                if not self.check_collision(self.active_tetromino, self.x, self.y + 1):
                    self.y += 1
                else:
                    self.place_active_tetromino()
                    return (
                        self._get_obs(),
                        REWARDS["game_over"] * self.clear_filled_rows()
                        + REWARDS["alife"],
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
        self.board = np.zeros((self.height, self.width), dtype=np.uint8)

        # Reset the randomizer
        self.randomizer.reset(seed=seed)

        # Get first piece from bag
        self.active_tetromino = self.tetrominoes[self.randomizer.get_next_tetromino()]

        self.x, self.y = (self.width // 2 - len(self.active_tetromino[0]) // 2, 0)
        self.game_over = False

        return self._get_obs(), {}

    def render(self) -> "RenderFrame | list[RenderFrame] | None":
        """Renders the environment as text."""
        if self.render_mode == "ansi":
            # Render active tetromino (because it's not on self.board)
            if self.active_tetromino is not None:
                tetromino_height, tetromino_width = self.active_tetromino.shape
                view = self.board.copy()
                view[
                    self.y : self.y + tetromino_height,
                    self.x : self.x + tetromino_width,
                ] += self.active_tetromino
            else:
                view = self.board

            char_field = np.where(view == 0, ".", view.astype(str))
            field_str = "\n".join("".join(row) for row in char_field)
            return field_str
        elif self.render_mode == "rgb_array":
            # Initialize rgb array
            rgb = np.zeros(
                (self.board.shape[0], self.board.shape[1], 3), dtype=np.uint8
            )
            # Display the board
            rgb[:, :, :] = STANDARD_COLORS[self.board]

            # Render active tetromino (because it's not on self.board)
            if self.active_tetromino is not None:
                inbound_tetromino, _ = self.cut_bounds(
                    self.x, self.y, self.active_tetromino
                )

                # Expand to 3 Dimensions
                active_tetromino_rgb = np.repeat(
                    inbound_tetromino[:, :, np.newaxis], 3, axis=2
                )
                active_tetromino_rgb[:, :, :] = STANDARD_COLORS[inbound_tetromino]

                # Apply by masking
                tetromino_height, tetromino_width = inbound_tetromino.shape
                x, y = max(0, self.x), max(0, self.y)
                rgb[
                    y : y + tetromino_height,
                    x : x + tetromino_width,
                ] = active_tetromino_rgb
            return rgb

        return None

    def spawn_tetromino(self):
        """Spawns a new tetromino at the top of the board.

        If the tetromino collides with the top of the board, the game is over.
        """
        self.active_tetromino = self.tetrominoes[self.randomizer.get_next_tetromino()]
        self.x, self.y = self.width // 2 - self.active_tetromino.shape[0] // 2, 0
        self.game_over = self.check_collision(self.active_tetromino, self.x, self.y)

    def place_active_tetromino(self):
        """Locks the active tetromino in place on the board."""
        # Boundary checks
        inbound_tetromino, _ = self.cut_bounds(self.x, self.y, self.active_tetromino)

        tetromino_height, tetromino_width = inbound_tetromino.shape
        x, y = max(0, self.x), max(0, self.y)
        self.board[
            y : y + tetromino_height, x : x + tetromino_width
        ] += inbound_tetromino
        self.active_tetromino = None

    def check_collision(self, tetromino: np.ndarray, x: int, y: int) -> bool:
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
        inbound_tetromino, outbound_tetromino = self.cut_bounds(x, y, tetromino)

        # Tetromino moving outside the board
        if (outbound_tetromino is not None) and np.any(outbound_tetromino != 0):
            return True

        # Extract the subarray of the board where the tetromino will be placed
        x, y = max(0, x), max(0, y)
        tetromino_height, tetromino_width = inbound_tetromino.shape
        board_subarray = self.board[y : y + tetromino_height, x : x + tetromino_width]

        # Check collision using numpy element-wise operations.
        # This checks if any corresponding cells (both non-zero) of the subarray and the
        # tetromino overlap, indicating a collision.
        return np.any(board_subarray[inbound_tetromino > 0] > 0)

    def cut_bounds(self, x: int, y: int, tetromino: np.ndarray):
        """Cut the tetromino to fit within the boundaries of the board.

        Args:
            x: The x position of the tetromino.
            y: The y position of the tetromino.
            tetromino: The tetromino to cut.

        Returns:
            Two new arrays. One is the part of the tetromino inside the board, and the other is the part outside.
        """
        # Default initial values
        inbound_tetromino = tetromino  # the part of the tetromino inside the board
        outbound_tetromino = None  # the part of the tetromino outside the board

        # Boundary checks
        tetromino_height, tetromino_width = tetromino.shape
        if x < 0:
            # Left overflow
            outbound_tetromino = tetromino[:, : abs(x)]
            inbound_tetromino = tetromino[:, abs(x) :]
        elif x + tetromino_width > self.width:
            # Right overflow
            outbound_tetromino = tetromino[:, self.width - x :]
            inbound_tetromino = tetromino[:, : self.width - x]
        if y + tetromino_height > self.height:
            # Bottom overflow
            outbound_tetromino = tetromino[self.height - y :, :]
            inbound_tetromino = tetromino[: self.height - y, :]

        return inbound_tetromino, outbound_tetromino

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
        while not self.check_collision(self.active_tetromino, self.x, self.y + 1):
            self.y += 1

    def clear_filled_rows(self) -> int:
        """Clear any filled rows on the board.

        The clearing is performed using numpy by indexing only the rows that are not filled and
        concatenating them with a new top part of the board that contains zeros.

        With this implementation, the clearing operation is efficient and does not require looping.

        Returns:
            The number of rows that were cleared.
        """
        # Check for filled rows. A row is filled if it does not contain any zeros.
        filled_rows = ~(self.board == 0).any(axis=1)
        num_filled = np.sum(filled_rows)

        if num_filled > 0:
            # Identify the rows that are not filled.
            unfilled_rows = self.board[~filled_rows]

            # Create a new top part of the board with zeros to compensate for the cleared rows.
            free_space = np.zeros((num_filled, self.width), dtype=np.uint8)

            # Concatenate the new top with the unfilled rows to form the updated board.
            self.board[:] = np.concatenate((free_space, unfilled_rows), axis=0)

        return num_filled

    def _get_obs(self) -> np.ndarray:
        """Return the current board as an observation."""
        return self.board.astype(np.float32)

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
