# Wrappers

Wrappers are used to extend or alter the functionality of an environment.
You can easily define your own observation wrappers by following the [Gymnasium documentation for wrappers](https://gymnasium.farama.org/api/wrappers/).
Alternatively, you can use predefined wrappers from Tetris Gymnasium or Gymnasium.

## Observation wrappers

Observation wrappers are used to alter the observation space of the environment. This can be useful for changing the
shape of the observation space or for adding additional information to the observation space.

### Implementations

```{eval-rst}
.. autoclass:: tetris_gymnasium.wrappers.observation.RgbObservation
```

#### Methods
```{eval-rst}
.. automethod:: tetris_gymnasium.wrappers.observation.RgbObservation.observation
.. automethod:: tetris_gymnasium.wrappers.observation.RgbObservation.render
```
