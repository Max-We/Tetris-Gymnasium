import sys

import cv2
import gymnasium as gym

from tetris_gymnasium.envs import Tetris

if __name__ == "__main__":
    # Create an instance of Tetris
    tetris_game = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    tetris_game.reset(seed=42)

    # Main game loop
    terminated = False
    while not terminated:
        # Render the current state of the game as text
        tetris_game.render()

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

            if (
                cv2.getWindowProperty(
                    tetris_game.unwrapped.window_name, cv2.WND_PROP_VISIBLE
                )
                == 0
            ):
                sys.exit()

        # Perform the action
        observation, reward, terminated, truncated, info = tetris_game.step(action)
        print(info)

    # Game over
    print("Game Over!")
