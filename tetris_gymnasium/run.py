import random

from tetris_gymnasium.envs.tetris import Tetris


if __name__ == '__main__':
    # Create an instance of Tetris
    tetris_game = Tetris()

    # Main game loop
    while not tetris_game.game_over:
        # Render the current state of the game as text
        tetris_game.render()

        # Take a random action (for demonstration purposes)
        action = random.randint(0, 5)

        # Perform the action
        tetris_game.step(action)

    # Game over
    print("Game Over!")
