import time

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from tetris_gymnasium.envs.tetris_fn import reset, step
from tetris_gymnasium.functional.core import EnvConfig, State
from tetris_gymnasium.functional.queue import (
    bag_queue_get_next_element,
    create_bag_queue,
)
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

ACTION_ID_TO_NAME = {
    0: "move_left",
    1: "move_right",
    2: "move_down",
    3: "rotate_clockwise",
    4: "rotate_counterclockwise",
    5: "do_nothing",
    6: "hard_drop",
}


class TetrisVisualizer:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.cell_size = 30
        self.window_width = self.config.width * self.cell_size
        # Add extra height for score display
        self.score_height = 40
        self.window_height = self.config.height * self.cell_size + self.score_height

        # Initialize game state
        self.key = jax.random.PRNGKey(0)
        self.key, self.state, self.observation = reset(
            TETROMINOES,
            self.key,
            self.config,
            create_bag_queue,
            bag_queue_get_next_element,
        )

        # Create color mapping array (piece_id -> RGB)
        # Index 0: empty cell (black)
        # Index 1: ghost piece if any (grey)
        # Indices 2+: tetromino colors from TETROMINOES
        self.colors = np.array(
            [[0, 0, 0]]
            + [[128, 128, 128]]
            + [tuple(map(int, color)) for color in TETROMINOES.colors]
        )

    def render(self) -> np.ndarray:
        # Create the score area (dark grey background)
        score_area = np.full(
            (self.score_height, self.window_width, 3), [30, 30, 30], dtype=np.uint8
        )

        # Add score text
        score_text = f"Score: {self.state.score}"
        cv2.putText(
            score_area,
            score_text,
            (10, 30),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Font scale
            (255, 255, 255),  # White color
            2,  # Thickness
        )

        # Convert observation to color indices and create game board
        board = np.array(self.observation)
        rgb_board = self.colors[board.astype(np.int32)]

        # Resize using nearest neighbor interpolation
        if self.cell_size > 1:
            rgb_board = rgb_board.repeat(self.cell_size, axis=0).repeat(
                self.cell_size, axis=1
            )

        # Add grid lines
        if self.cell_size >= 10:  # Only add grid if cells are large enough
            # Vertical lines
            rgb_board[:, :: self.cell_size] = [50, 50, 50]
            # Horizontal lines
            rgb_board[:: self.cell_size, :] = [50, 50, 50]

        # Add game over text if needed
        if self.state.game_over:
            # Calculate text position
            text_img = np.zeros_like(rgb_board)
            text_pos = (
                self.window_width // 4,
                self.config.height * self.cell_size // 2,
            )
            cv2.putText(
                text_img,
                "GAME OVER",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            # Overlay text
            mask = text_img.any(axis=2)
            rgb_board[mask] = text_img[mask]

        # Combine score area and game board
        full_display = np.vstack([score_area, rgb_board])
        return full_display

    def process_action(self, action: int) -> bool:
        if self.state.game_over:
            return False

        self.key, self.state, self.observation, reward, done, info = step(
            TETROMINOES,
            self.key,
            self.state,
            action,
            self.config,
            bag_queue_get_next_element,
        )

        return not done

    def reset_game(self):
        self.key = jax.random.PRNGKey(int(time.time()))
        self.key, self.state, self.observation = reset(
            TETROMINOES,
            self.key,
            self.config,
            create_bag_queue,
            bag_queue_get_next_element,
        )


def main():
    config = EnvConfig(
        width=10, height=20, padding=10, queue_size=7, gravity_enabled=False
    )
    game = TetrisVisualizer(config)

    cv2.namedWindow("Tetris")

    while True:
        # Render and display
        frame = game.render()
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow("Tetris", frame_bgr)

        # Wait for keypress
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            game.reset_game()
        elif key == 32:  # Space
            game.process_action(6)  # hard_drop
        elif key == ord("z"):
            game.process_action(4)  # rotate_counterclockwise
        elif key == 81 or key == ord("a"):  # Left arrow or 'a'
            game.process_action(0)  # move_left
        elif key == 83 or key == ord("d"):  # Right arrow or 'd'
            game.process_action(1)  # move_right
        elif key == 82 or key == ord("w"):  # Up arrow or 'w'
            game.process_action(3)  # rotate_clockwise
        elif key == 84 or key == ord("s"):  # Down arrow or 's'
            game.process_action(2)  # move_down
        else:
            game.process_action(5)  # do_nothing

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
