---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/logo.png
:alt: Tetris Gymnasium Logo
```

```{project-heading}
A customisable, easy-to-use and performant Tetris environment for Gymnasium
```

Tetris Gymnasium is tightly integrated with Gymnasium and exposes a simple API for training agents to play Tetris.

The environment offers state-of-the-art performance and holds a high standard for code quality. With it, researchers and developers can focus on their research and development, rather than the environment itself.

Getting started is easy. Here is a simple example of an environment with random actions:

```{code-block} python
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

```{toctree}
:maxdepth: 2
:caption: Introduction
:hidden:

introduction/installation
introduction/quickstart
```

```{toctree}
:maxdepth: 2
:caption: API
:hidden:

environments/tetris
```

```{toctree}
:maxdepth: 2
:caption: Development
:hidden:

development/contributing
```
