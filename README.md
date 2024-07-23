![logo](./docs/_static/logo.png "Tetris Gymnasium")

Tetris Gymnasium is tightly integrated with Gymnasium and exposes a simple API for training agents to play Tetris.

Getting started is easy. Here is a simple example of an environment with random actions:

```python
import gymnasium as gym
from tetris_gymnasium.envs import Tetris

env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
env.reset(seed=42)

terminated = False
while not terminated:
    print(env.render() + "\n")
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
print("Game Over!")
```

## Background

Tetris Gymnasium tries to solve problems of other environments by being modular, understandable and adjustable. You can read more about the background in our paper: _Piece by Piece: Assembling a Modular Reinforcement Learning Environment for Tetris_ ([Preprint on EasyChair](https://easychair.org/publications/preprint/154Q)).

Abstract:

>The game of Tetris is an open challenge in machine learning and especially Reinforcement Learning (RL). Despite its popularity, contemporary environments for the game lack key qualities, such as a clear documentation, an up-to-date codebase or game related features.
This work introduces Tetris Gymnasium, a modern RL environment built with Gymnasium, that aims to address these problems by being modular, understandable and adjustable.
To evaluate Tetris Gymnasium on these qualities, a Deep Q Learning agent was trained and compared to a baseline environment, and it was found that it fulfills all requirements of a feature-complete RL environment while being adjustable to many different requirements.
The source-code and documentation is available at on GitHub and can be used for free under the MIT license.

## Documentation

The full documentation of the project can be found on [GitHub Pages](https://max-we.github.io/Tetris-Gymnasium/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank the creators and maintainers of [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), [CleanRL](https://github.com/vwxyzjn/cleanrl) and [Tetris-deep-Q-learning-pytorch](https://github.com/uvipen/Tetris-deep-Q-learning-pytorch) for providing a powerful frameworks and reference implementations.
