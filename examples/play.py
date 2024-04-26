import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    # Create an instance of Tetris
    tetris_game = gym.make("tetris_gymnasium/Tetris")
    tetris_game.reset(seed=42)

    # Main game loop
    while not tetris_game.unwrapped.game_over:
        # Render the current state of the game as text
        tetris_game.render()

        # Take a random action (for demonstration purposes)
        action = tetris_game.action_space.sample()

        # Perform the action
        tetris_game.step(action)

    # Game over
    print("Game Over!")
