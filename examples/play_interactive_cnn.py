import sys

import cv2
import gymnasium as gym

from tetris_gymnasium.envs import Tetris
from tetris_gymnasium.wrappers.observation import CnnObservation

if __name__ == "__main__":
    # Create an instance of Tetris
    tetris_game = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    tetris_game.reset(seed=42)
    tetris_game = CnnObservation(tetris_game)

    window_name = "Tetris Gymnasium"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, 395, 250)

    # Main game loop
    terminated = False
    while not terminated:
        # Render the current state of the game as text
        rgb = tetris_game.render()

        # Render the current state of the game as an image using CV2
        # CV2 uses BGR color format, so we need to convert the RGB image to BGR
        cv2.imshow(window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(50)

        # Pick an action from user input mapped to the keyboard
        action = None
        while action is None:
            key = cv2.waitKey(1)

            if key == ord("a"):
                action = tetris_game.unwrapped.actions.move_left
            elif key == ord("d"):
                action = tetris_game.unwrapped.actions.move_right
            elif key == ord("s"):
                action = tetris_game.unwrapped.actions.move_down
            elif key == ord("w"):
                action = tetris_game.unwrapped.actions.rotate_counterclockwise
            elif key == ord("e"):
                action = tetris_game.unwrapped.actions.rotate_clockwise
            elif key == ord(" "):
                action = tetris_game.unwrapped.actions.hard_drop
            elif key == ord("q"):
                action = tetris_game.unwrapped.actions.swap
            elif key == ord("r"):
                tetris_game.reset(seed=42)
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 0:
                sys.exit()

        # Perform the action
        observation, reward, terminated, truncated, info = tetris_game.step(action)

    # Game over
    print("Game Over!")
