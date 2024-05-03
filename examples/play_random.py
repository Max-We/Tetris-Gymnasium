import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    # Create an instance of Tetris
    tetris_game = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    tetris_game.reset(seed=42)

    # Main game loop
    terminated = False
    while not terminated:
        # Render the current state of the game as text
        ansi = tetris_game.render()
        print(ansi + "\n")

        # Take a random action (for demonstration purposes)
        action = tetris_game.action_space.sample()

        # Perform the action
        observation, reward, terminated, truncated, info = tetris_game.step(action)

    # Game over
    print("Game Over!")
