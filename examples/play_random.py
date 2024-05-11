import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    tetris_game = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    tetris_game.reset(seed=42)

    terminated = False
    while not terminated:
        ansi = tetris_game.render()
        print(ansi + "\n")

        action = tetris_game.action_space.sample()

        observation, reward, terminated, truncated, info = tetris_game.step(action)
    print("Game Over!")
