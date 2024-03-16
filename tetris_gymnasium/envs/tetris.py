from typing import Any, List

import numpy as np
import gymnasium as gym
from gymnasium.core import RenderFrame, ActType
from gymnasium.spaces import Discrete, Box

from tetris_gymnasium.components.scheduler import BagScheduler, Scheduler
from tetris_gymnasium.util.tetrominoes import STANDARD_TETROMINOES

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
    def __init__(self, render_mode=None, width=10, height=20, tetrominoes=STANDARD_TETROMINOES, scheduler=BagScheduler):
        # Dimensions
        self.height: int = height
        self.width: int = width
        self.window_width: int = width * 100
        self.window_height: int = height * 100

        # Tetrominoes & Schedule
        self.scheduler: Scheduler = scheduler(len(tetrominoes))
        self.tetrominoes: List[np.ndarray] = tetrominoes
        self.active_tetromino: np.ndarray = self.tetrominoes[self.scheduler.get_next_tetromino()]
        self.board: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)

        # Position
        self.x: int = 0
        self.y: int = 0

        # Game state
        self.game_over: bool = False

        # Gymnasium
        self.observation_space = Box(low=0, high=len(self.tetrominoes), shape=(width, height), dtype=np.int8)
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

    def step(self, action: ActType) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        https://gymnasium.farama.org/api/env/#gymnasium.Env.step
        """
        # Todo: impl. truncation
        truncated = False

        if self.active_tetromino is None:
            self.spawn_tetromino()
        else:
            if action == ACTIONS["move_left"]:  # move left
                if not self.check_collision(self.active_tetromino, self.x - 1, self.y):
                    self.x -= 1
            elif action == ACTIONS["move_left"]:  # move right
                if not self.check_collision(self.active_tetromino, self.x + 1, self.y):
                    self.x += 1
            elif action == ACTIONS["rotate_clockwise"]:  # rotate clockwise
                if not self.check_collision(self.rotate(self.active_tetromino, True), self.x, self.y):
                    self.active_tetromino = self.rotate(self.active_tetromino, True)
            elif action == ACTIONS["rotate_counterclockwise"]:  # rotate counterclockwise
                if not self.check_collision(self.rotate(self.active_tetromino, False), self.x, self.y):
                    self.active_tetromino = self.rotate(self.active_tetromino, False)
            elif action == ACTIONS["hard_drop"]:  # hard drop
                self.drop_active_tetromino()
                self.place_active_tetromino()
                return self.board, REWARDS["game_over"] * self.clear_filled_rows() + REWARDS["alife"], False, truncated, {}
            elif action == ACTIONS["move_down"]:  # move down (gravity)
                # Todo: Separate move_down from gravity
                if not self.check_collision(self.active_tetromino, self.x, self.y + 1):
                    self.y += 1
                else:
                    self.place_active_tetromino()
                    return self.board, REWARDS["game_over"] * self.clear_filled_rows() + REWARDS[
                        "alife"], False, truncated, {}

        if self.game_over:
            return self.board, REWARDS["game_over"], True, truncated, {}

        return self.board, REWARDS["alife"], False, truncated, {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
            np.ndarray, dict[str, Any]]:
        # Initialize fresh board
        self.board = np.zeros((self.height, self.width), dtype=np.int8)

        # Create bag
        self.scheduler = BagScheduler(len(self.tetrominoes))

        # Get first piece from bag
        self.active_tetromino = self.tetrominoes[self.scheduler.get_next_tetromino()]

        self.x, self.y = (self.width // 2 - len(self.active_tetromino[0]) // 2, 0)
        self.game_over = False

        return self.board, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.active_tetromino is not None:
            tetromino_height, tetromino_width = self.active_tetromino.shape
            view = self.board.copy()
            view[self.y:self.y + tetromino_height, self.x:self.x + tetromino_width] += self.active_tetromino
        else:
            view = self.board

        char_field = np.where(view == 0, '.', view.astype(str))
        field_str = '\n'.join(''.join(row) for row in char_field)
        print(field_str)
        print("==========")

        return None

    def spawn_tetromino(self):
        self.active_tetromino = self.tetrominoes[self.scheduler.get_next_tetromino()]
        self.x, self.y = self.width // 2 - self.active_tetromino.shape[0] // 2, 0
        self.game_over = self.check_collision(self.active_tetromino, self.x, self.y)

    def place_active_tetromino(self):
        tetromino_height, tetromino_width = self.active_tetromino.shape
        self.board[self.y:self.y + tetromino_height, self.x:self.x + tetromino_width] += self.active_tetromino
        self.active_tetromino = None

    def check_collision(self, tetromino: np.ndarray, x: int, y: int) -> bool:
        tetromino_height, tetromino_width = tetromino.shape

        # Boundary check
        if (x < 0 or x + tetromino_width > self.width) or (y < 0 or y + tetromino_height > self.height):
            return True

        # Extract the subarray of the board where the tetromino will be placed
        board_subarray = self.board[y:y + tetromino_height, x:x + tetromino_width]

        # Check collision using numpy element-wise operations.
        # This checks if any corresponding cells (both non-zero) of the subarray and the
        # tetromino overlap, indicating a collision.
        if np.any(board_subarray[tetromino > 0] > 0):
            return True

        # No collision detected
        return False

    def rotate(self, tetromino: np.ndarray, clockwise=True) -> np.ndarray:
        return np.rot90(tetromino, k=(1 if clockwise else -1))

    def drop_active_tetromino(self):
        # Todo: is there a more efficient way to hard drop without loop?
        while not self.check_collision(self.active_tetromino, self.x, self.y + 1):
            self.y += 1

    def clear_filled_rows(self) -> int:
        # Check for filled rows. A row is filled if it does not contain any zeros.
        filled_rows = ~(self.board == 0).any(axis=1)
        num_filled = np.sum(filled_rows)

        if num_filled > 0:
            # Identify the rows that are not filled.
            unfilled_rows = self.board[~filled_rows]

            # Create a new top part of the board with zeros to compensate for the cleared rows.
            free_space = np.zeros((num_filled, self.width), dtype=np.int8)

            # Concatenate the new top with the unfilled rows to form the updated board.
            self.board[:] = np.concatenate((free_space, unfilled_rows), axis=0)

        return num_filled

    # def close(self):
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
