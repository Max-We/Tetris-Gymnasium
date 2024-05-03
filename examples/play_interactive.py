import sys

import cv2
import gymnasium as gym

from tetris_gymnasium.envs.tetris import ACTIONS

if __name__ == "__main__":
    # Create an instance of Tetris
    tetris_game = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    tetris_game.reset(seed=42)

    cv2.namedWindow("Tetris", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("Tetris", 200, 400)

    # Main game loop
    terminated = False
    while not terminated:
        # Render the current state of the game as text
        rgb = tetris_game.render()

        # Render the current state of the game as an image using CV2
        # CV2 uses BGR color format, so we need to convert the RGB image to BGR
        cv2.imshow("Tetris", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(50)

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
        observation, reward, terminated, truncated, info = tetris_game.step(action)

    # Game over
    print("Game Over!")
