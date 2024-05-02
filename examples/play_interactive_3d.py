import sys

import cv2
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from tetris_gymnasium.envs.tetris import Tetris

ACTIONS = {
    "move_left": 3,
    "move_right": 2,
    "move_down": 0,
    "move_up": 1,
    "rotate_clockwise": 4,
    "rotate_counterclockwise": 5,
    "hard_drop": 6,
}

if __name__ == "__main__":
    # Create an instance of Tetris
    tetris_game = gym.make(
        "tetris_gymnasium/Tetris", render_mode="rgb_array", dimensions=(20, 10, 3)
    )
    tetris_game.reset(seed=42)

    cv2.namedWindow("Tetris", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("Tetris", 200, 400)

    # Main game loop
    while not tetris_game.unwrapped.game_over:
        # Render the current state of the game as text
        rgb = tetris_game.render()

        # Setup the plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Prepare data for scatter plotting
        x = np.arange(rgb.shape[1])  # Width of the board (columns)
        y = np.arange(rgb.shape[0])  # Height of the board (rows)
        z = np.arange(rgb.shape[2])  # Depth (layers, each having an RGB)

        # Meshgrid for coordinates in a 3D space
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        # Flatten arrays for matplotlib
        xx = xx.flatten()
        yy = yy.flatten()
        zz = zz.flatten()

        # Colors need to be reshaped to match coordinates and normalized
        colors = rgb / 255.0
        colors = colors.reshape(-1, 3)  # Flatten to match coordinate points

        # Plot each point with its corresponding color
        ax.scatter(xx, yy, zz, c=colors, marker="s")

        # Labeling Axes
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")
        ax.set_zlabel("Layer Index")

        # Show the plot
        plt.show()

        # Pick an action from user input mapped to the keyboard
        action = None
        while action is None:
            key = cv2.waitKey(1)

            if key == ord("a"):
                action = ACTIONS["move_left"]
            elif key == ord("d"):
                action = ACTIONS["move_right"]
            elif key == ord("s"):
                action = ACTIONS["move_down"]
            elif key == ord("w"):
                action = ACTIONS["rotate_counterclockwise"]
            elif key == ord("q"):
                action = ACTIONS["rotate_clockwise"]
            elif key == ord(" "):
                action = ACTIONS["hard_drop"]
            elif key == ord("r"):
                tetris_game.reset(seed=42)
                break

            if cv2.getWindowProperty("Tetris", cv2.WND_PROP_VISIBLE) == 0:
                sys.exit()

        # Perform the action
        tetris_game.step(action)

    # Game over
    print("Game Over!")
