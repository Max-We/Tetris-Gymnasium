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
        self.window_height = self.config.height * self.cell_size

        # Initialize game state
        self.key = jax.random.PRNGKey(0)
        self.key, self.state, self.observation = reset(
            TETROMINOES,
            self.key,
            self.config,
            create_bag_queue,
            bag_queue_get_next_element,
        )

    def rgb_to_bgr(self, rgb_color):
        """Convert RGB color tuple to BGR for cv2"""
        return (rgb_color[2], rgb_color[1], rgb_color[0])

    def render(self) -> np.ndarray:
        image = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)

        # Convert JAX observation to numpy
        board = np.array(self.observation)

        # Draw the board
        for i in range(self.config.height):
            for j in range(self.config.width):
                if board[i, j] > 0:
                    # Get color from TETROMINOES colors and convert to BGR
                    rgb_color = tuple(
                        map(int, TETROMINOES.colors[board[i, j] - 2])
                    )  # -2 because piece IDs start at 2
                    bgr_color = self.rgb_to_bgr(rgb_color)
                    cv2.rectangle(
                        image,
                        (j * self.cell_size, i * self.cell_size),
                        ((j + 1) * self.cell_size, (i + 1) * self.cell_size),
                        bgr_color,
                        -1,
                    )

        # Draw grid lines
        grid_color = self.rgb_to_bgr((50, 50, 50))
        text_color = self.rgb_to_bgr((255, 255, 255))

        for i in range(self.config.height + 1):
            cv2.line(
                image,
                (0, i * self.cell_size),
                (self.window_width, i * self.cell_size),
                grid_color,
                1,
            )
        for j in range(self.config.width + 1):
            cv2.line(
                image,
                (j * self.cell_size, 0),
                (j * self.cell_size, self.window_height),
                grid_color,
                1,
            )

        if self.state.game_over:
            cv2.putText(
                image,
                "GAME OVER",
                (self.window_width // 4, self.window_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color,
                2,
            )

        return image

    def process_action(self, action: int) -> bool:
        if self.state.game_over:
            return False

        # Use the existing step function
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
        self.key = jax.random.PRNGKey(int(time.time()))  # New random seed
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
        cv2.imshow("Tetris", frame)

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
    # disable jit jax
    # jax.config.update("jax_disable_jit", True)

    main()
