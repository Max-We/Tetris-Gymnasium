# Quickstart

After completing the [Installation](installation.md), you can start using the environment by importing it in your Python code and calling the `gymnasium.make` function.

```{code-block} python
import gymnasium as gym
from tetris_gymnasium.envs import Tetris

env = gym.make("tetris_gymnasium/Tetris")
```

With the environment created, you can interact with it by calling the Gymnasium typical `reset` and `step` methods. The `reset` method initializes the environment and returns the initial observation and info. The `step` method takes an action as input and returns the next observation, reward, termination flag, truncation flag, and info.

## Simple random agent

For example, a simple loop that interacts with the environment using random actions could look like this:

```{eval-rst}
.. literalinclude:: ../../examples/play_random.py
    :language: python
```

## Interactive environment

You can play around with the environment by using the interactive scripts in the `examples` directory.

For example, the `play_interactive.py` script allows you to play the Tetris environment using the keyboard.

```{eval-rst}
.. literalinclude:: ../../examples/play_interactive.py
    :language: python
```


## Training

In order to do Reinforcement Learning, you need to train an agent. To give some real-world examples, the code in the files `train_lin.py` and `train_cnn.py` in the `examples` directory show how to train a DQN agent on the Tetris environment.

To run the training, you can use the following command:

```{code-block} bash
poetry run python examples/train_lin.py # uses a linear model
# or
poetry run python examples/train_cnn.py # uses convolutions
```

You can refer to the [CleanRL documentation](https://docs.cleanrl.dev/rl-algorithms/dqn/) for more information on the training script. Note: If you have tracking enabled, you will be prompted to login to Weights & Biases during the first run. This behavior can be adjusted in the script.
