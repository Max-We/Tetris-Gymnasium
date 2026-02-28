# Tetris (JAX)

| Description       | Details                                              |
|-------------------|------------------------------------------------------|
| Action Space      | `int` in `[0, 6]`                                    |
| Observation Space | `jax.Array` of shape `(height, width)`, dtype `int8` |
| Import            | `from tetris_gymnasium.envs.tetris_fn import reset, step` |

## Description

The functional environment is a **pure-function, JAX-native** implementation of Tetris.
Unlike the Gymnasium-based environment, it has no class hierarchy, no `gym.Env` API,
and no internal mutable state.
Every operation is a stateless function that takes the current state as input and
returns a new state — making it fully compatible with `jax.jit`, `jax.vmap`, and
`jax.grad`.

Key benefits over the standard environment:

- **JIT compilation** — wrap `reset` or `step` with `jax.jit` for fast repeated calls.
- **Vectorised environments** — use `batched_reset` / `batched_step` (or `jax.vmap`)
  to run thousands of environments in parallel on a single device.
- **Functional purity** — no hidden state; reproducibility is guaranteed by the PRNG key.
- **No Gymnasium dependency** — suitable for JAX-only training pipelines.

## Configuration

The environment is configured through an `EnvConfig` named tuple:

```{eval-rst}
.. autoclass:: tetris_gymnasium.functional.core.EnvConfig
   :members:
   :no-index:
```

Default values used throughout this page:

```python
from tetris_gymnasium.functional.core import EnvConfig

config = EnvConfig(width=10, height=20, padding=4, queue_size=7)
```

## State

All game state is held in a `State` dataclass:

```{eval-rst}
.. autoclass:: tetris_gymnasium.functional.core.State
   :members:
```

| Field               | Shape  | Description                                    |
|---------------------|--------|------------------------------------------------|
| `board`             | (H, W) | Padded board; bedrock padding has value 1      |
| `active_tetromino`  | ()     | Index into the tetrominoes array (0–6)         |
| `rotation`          | ()     | Current rotation of the active piece (0–3)     |
| `x`                 | ()     | Column position of the active piece            |
| `y`                 | ()     | Row position of the active piece               |
| `queue`             | (L,)   | Piece queue (indices into tetrominoes array)   |
| `queue_index`       | ()     | Current position in the queue                  |
| `game_over`         | ()     | Boolean flag                                   |
| `score`             | ()     | Cumulative score                               |
| `rng_key`           | (2,)   | JAX PRNG key for internal randomness           |

## Tetrominoes

The standard set of seven tetrominoes is provided as a pre-built `TETROMINOES` constant:

```python
from tetris_gymnasium.functional.tetrominoes import TETROMINOES
```

Pass this object to `reset`, `step`, and all core functions.

## Basic Usage

### reset

```python
import jax
from tetris_gymnasium.envs.tetris_fn import reset
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

config = EnvConfig(width=10, height=20, padding=4, queue_size=7)
key = jax.random.PRNGKey(42)

key, state, observation = reset(TETROMINOES, key, config)
```

`reset` returns a **3-tuple**:

| Return value  | Type         | Description                                      |
|---------------|--------------|--------------------------------------------------|
| `key`         | `PRNGKey`    | Updated PRNG key (pass to the next call)         |
| `state`       | `State`      | Initial game state                               |
| `observation` | `jax.Array`  | Initial observation of shape `(height, width)`   |

### step

```python
from tetris_gymnasium.envs.tetris_fn import step

action = 0  # move left
state, observation, reward, terminated, info = step(
    TETROMINOES, state, action, config
)
```

`step` returns a **5-tuple**:

| Return value  | Type        | Description                                         |
|---------------|-------------|-----------------------------------------------------|
| `state`       | `State`     | Updated game state                                  |
| `observation` | `jax.Array` | New observation of shape `(height, width)`          |
| `reward`      | `float`     | Reward for this step (`new_score − old_score`)      |
| `terminated`  | `bool`      | `True` when the game is over                        |
| `info`        | `dict`      | `{"lines_cleared": int}`                            |

When `terminated` is `True`, subsequent calls to `step` are no-ops — state and
observation are returned unchanged with zero reward.

### Random agent example

```{eval-rst}
.. literalinclude:: ../../examples/play_random_functional.py
   :language: python
```

## Actions

| ID | Name                   | Effect                                           |
|----|------------------------|--------------------------------------------------|
| 0  | `move_left`            | Move active piece one column left                |
| 1  | `move_right`           | Move active piece one column right               |
| 2  | `move_down`            | Move active piece one row down (soft drop)       |
| 3  | `rotate_counterclockwise` | Rotate active piece 90° counter-clockwise     |
| 4  | `rotate_clockwise`     | Rotate active piece 90° clockwise                |
| 5  | `do_nothing`           | No movement; gravity still applies if enabled    |
| 6  | `hard_drop`            | Drop piece instantly to lowest valid position    |

Moves that would result in a collision are silently ignored (piece stays in place).

## Rewards

| Event              | Reward formula            |
|--------------------|---------------------------|
| Lines cleared      | `max(lines * 200 − 100, 0)` |
| Tetris (4 lines)   | 800 (flat bonus)          |
| Hard drop distance | `2 × cells dropped`       |

The reward returned by `step` is the **delta** of `state.score` between steps, so it
already combines all sources for that step.

## Observation

The observation is a 2D integer array of shape `(height, width)` (padding stripped):

| Value | Meaning                        |
|-------|--------------------------------|
| `0`   | Empty cell                     |
| `1`   | Locked piece                   |
| `-1`  | Active (falling) piece         |

```{eval-rst}
.. autofunction:: tetris_gymnasium.envs.tetris_fn.get_observation
```

## JIT Compilation

Wrap `reset` and `step` with `jax.jit` to compile them once and execute fast:

```python
import jax
from functools import partial
from tetris_gymnasium.envs.tetris_fn import reset, step
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

config = EnvConfig(width=10, height=20, padding=4, queue_size=7)

jit_reset = jax.jit(partial(reset, TETROMINOES, config=config))
jit_step  = jax.jit(partial(step,  TETROMINOES, config=config))

key = jax.random.PRNGKey(0)
key, state, obs = jit_reset(key)
state, obs, reward, terminated, info = jit_step(state, 6)
```

## Batched (Vectorised) Environments

Run multiple independent environments in parallel using `batched_reset` and
`batched_step`.
These functions use `jax.vmap` internally and are JIT-compiled automatically.

```python
import jax
import jax.numpy as jnp
from tetris_gymnasium.envs.tetris_fn import batched_reset, batched_step
from tetris_gymnasium.functional.core import EnvConfig
from tetris_gymnasium.functional.tetrominoes import TETROMINOES

config = EnvConfig(width=10, height=20, padding=4, queue_size=7)
BATCH = 64

keys = jax.random.split(jax.random.PRNGKey(0), BATCH)
keys, states, observations = batched_reset(
    TETROMINOES, keys, config=config, batch_size=BATCH
)

# All environments take a different action
actions = jnp.zeros(BATCH, dtype=jnp.int32)
states, observations, rewards, terminated, info = batched_step(
    TETROMINOES, states, actions, config=config
)
```

All state fields gain a leading batch dimension of size `BATCH`.

```{eval-rst}
.. autofunction:: tetris_gymnasium.envs.tetris_fn.batched_reset
```

```{eval-rst}
.. autofunction:: tetris_gymnasium.envs.tetris_fn.batched_step
```

## API Reference

```{eval-rst}
.. autofunction:: tetris_gymnasium.envs.tetris_fn.reset
```

```{eval-rst}
.. autofunction:: tetris_gymnasium.envs.tetris_fn.step
```

### Core functions

```{eval-rst}
.. autofunction:: tetris_gymnasium.functional.core.create_board
.. autofunction:: tetris_gymnasium.functional.core.collision
.. autofunction:: tetris_gymnasium.functional.core.hard_drop
.. autofunction:: tetris_gymnasium.functional.core.lock_active_tetromino
.. autofunction:: tetris_gymnasium.functional.core.clear_filled_rows
.. autofunction:: tetris_gymnasium.functional.core.check_game_over
.. autofunction:: tetris_gymnasium.functional.core.score
```
