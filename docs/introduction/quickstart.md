# Quickstart

After completing the installation, you can start using the library by importing it in your Python code and calling the `gym.make` function with the name of the environment you want to use.

```{code-block} python
import gymnasium as gym
from tetris_gymnasium.envs import Tetris

env = gym.make("tetris_gymnasium/Tetris")
```

With the environment created, you can interact with it by calling the Gymnasium typical `reset` and `step` methods. The `reset` method initializes the environment and returns the initial observation and info. The `step` method takes an action as input and returns the next observation, reward, termination flag, truncation flag, and info.

## Random agent

For example, a simple loop that interacts with the environment using random actions could look like this:

> Note: Currently there is only a text renderer available. A graphical renderer is in development.

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

## Training a DQN agent

In order to do Reinforcement Learning, you need to train an agent. To give some real-world examples, the code in the files `train_lin.py` and `train_cnn.py` in the `examples` directory show how to train a DQN agent on the Tetris environment.

To run the training, you can use the following command:

```{code-block} bash
poetry run python examples/train_lin.py # uses a linear model
# or
poetry run python examples/train_cnn.py # uses convolutions
```
