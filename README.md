![logo](./docs/_static/logo.png "Tetris Gymnasium")

> Tetris Gymnasium is currently under early development!

Tetris Gymnasium is tightly integrated with Gymnasium and exposes a simple API for training agents to play Tetris.

The environment offers state-of-the-art performance and holds a high standard for code quality. With it, researchers and developers can focus on their research and development, rather than the environment itself.

Getting started is easy. Here is a simple example of an environment with random actions:

```python
import gymnasium as gym
from tetris_gymnasium.envs import Tetris

env = gym.make("tetris_gymnasium/Tetris")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
```

##

The documentation can be found on [GitHub Pages](https://max-we.github.io/Tetris-Gymnasium/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank the creators and maintainers of Gymnasium, CleanRL and Tetris-deep-Q-learning-pytorch for providing a powerful frameworks and reference implementations

---

Enjoy using the Gymnasium Tetris Environment for your reinforcement learning experiments! If you have any questions or need further assistance, don't hesitate to reach out to us. Happy coding! 🎮🕹️
