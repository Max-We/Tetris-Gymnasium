[![Python](https://img.shields.io/pypi/pyversions/gymnasium.svg)](https://badge.fury.io/py/tetris-gymnasium)
[![PyPI](https://badge.fury.io/py/gymnasium.svg)](https://badge.fury.io/py/tetris-gymnasium)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Tetris Gymnasium

![logo](https://raw.githubusercontent.com/Max-We/Tetris-Gymnasium/main/docs/_static/logo.png "Tetris Gymnasium")

Tetris Gymnasium is a state-of-the-art, modular Reinforcement Learning (RL) environment for Tetris, tightly integrated
with OpenAI's Gymnasium.

## Quick Start

Getting started with Tetris Gymnasium is straightforward. Here's an example to run an environment with random
actions:

```python
import cv2
import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)

    terminated = False
    while not terminated:
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        key = cv2.waitKey(100) # timeout to see the movement
    print("Game Over!")

```

For more examples, e.g. training a DQN agent, please refer to the [examples](examples) directory.

## Installation

Tetris Gymnasium can be installed via pip:

```bash
pip install tetris-gymnasium
```

## Why Tetris Gymnasium?

While significant progress has been made in RL for many Atari games, Tetris remains a challenging problem for AI, similar
to games like Pitfall. Its combination of NP-hard complexity, stochastic elements, and the need for long-term planning
makes it a persistent open problem in RL research. Tetris's intuitive gameplay and relatively modest computational
requirements position it as a potentially useful environment for developing and evaluating RL approaches in a demanding
setting.

Tetris Gymnasium aims to provide researchers and developers with a tool to address this challenge:

1. **Modularity**: The environment's architecture allows for customization and extension, facilitating exploration of
   various RL techniques.
2. **Clarity**: Comprehensive documentation and a structured codebase are designed to enhance accessibility and support
   experimentation.
3. **Adjustability**: Configuration options enable researchers to focus on specific aspects of the Tetris challenge as
   needed.
4. **Up-to-date**: Built on the current Gymnasium framework, the environment is compatible with contemporary RL
   algorithms and tools.
5. **Feature-rich**: Includes game-specific features that are sometimes absent in other Tetris environments, aiming to
   provide a more comprehensive representation of the game's challenges.

These attributes make Tetris Gymnasium a potentially useful resource for both educational purposes and RL research. By
providing a standardized yet adaptable platform for approaching one of RL's ongoing challenges, Tetris Gymnasium may
contribute to further exploration and development in Tetris RL.

## Documentation

For detailed information on using and customizing Tetris Gymnasium, please refer to
our [full documentation](https://max-we.github.io/Tetris-Gymnasium/).

## Background

Tetris Gymnasium addresses the limitations of existing Tetris environments by offering a modular, understandable, and
adjustable platform. Our paper, "Piece by Piece: Assembling a Modular Reinforcement Learning Environment for Tetris,"
provides an in-depth look at the motivations and design principles behind this project.

**Abstract:**

> The game of Tetris is an open challenge in machine learning and especially Reinforcement Learning (RL). Despite its
> popularity, contemporary environments for the game lack key qualities, such as a clear documentation, an up-to-date
> codebase or game related features. This work introduces Tetris Gymnasium, a modern RL environment built with
> Gymnasium,
> that aims to address these problems by being modular, understandable and adjustable. To evaluate Tetris Gymnasium on
> these qualities, a Deep Q Learning agent was trained and compared to a baseline environment, and it was found that it
> fulfills all requirements of a feature-complete RL environment while being adjustable to many different requirements.
> The source-code and documentation is available at on GitHub and can be used for free under the MIT license.

Read the full paper: [Preprint on EasyChair](https://easychair.org/publications/preprint/154Q)

## Citation

If you use Tetris Gymnasium in your research, please cite our work:

```bibtex
@booklet{EasyChair:13437,
  author    = {Maximilian Weichart and Philipp Hartl},
  title     = {Piece by Piece: Assembling a Modular Reinforcement Learning Environment for Tetris},
  howpublished = {EasyChair Preprint 13437},
  year      = {EasyChair, 2024}}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We extend our gratitude to the creators and maintainers
of [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), [CleanRL](https://github.com/vwxyzjn/cleanrl),
and [Tetris-deep-Q-learning-pytorch](https://github.com/uvipen/Tetris-deep-Q-learning-pytorch) for providing powerful
frameworks and reference implementations that have contributed to the development of Tetris Gymnasium.
