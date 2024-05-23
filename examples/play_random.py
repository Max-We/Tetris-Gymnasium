import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    env.reset(seed=42)

    terminated = False
    while not terminated:
        print(env.render() + "\n")
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
    print("Game Over!")
