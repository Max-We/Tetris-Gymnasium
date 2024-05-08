# Randomizer

```{eval-rst}
.. autoclass:: tetris_gymnasium.components.tetromino_randomizer.Randomizer
```

## Methods
```{eval-rst}
.. automethod:: tetris_gymnasium.components.tetromino_randomizer.Randomizer.get_next_tetromino
.. automethod:: tetris_gymnasium.components.tetromino_randomizer.Randomizer.reset
```

## Implementations

In Tetris Gymnasium, there are different randomizers available by default. The default randomizer is the `BagRandomizer`,
which is the same as the one used in the most Tetris games. The `TrueRandomizer` is a randomizer that generates
tetrominoes with a uniform distribution.

If these randomizers do not fit your needs, you can easily implement your own randomizer by subclassing the `Randomizer`.

```{eval-rst}
.. autoclass:: tetris_gymnasium.components.tetromino_randomizer.BagRandomizer
```

```{eval-rst}
.. automethod:: tetris_gymnasium.components.tetromino_randomizer.BagRandomizer.get_next_tetromino
```

```{eval-rst}
.. autoclass:: tetris_gymnasium.components.tetromino_randomizer.TrueRandomizer
```

```{eval-rst}
.. automethod:: tetris_gymnasium.components.tetromino_randomizer.TrueRandomizer.get_next_tetromino
```
