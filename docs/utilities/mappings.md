# Mappings

Mappings are small dataclasses that map the integer values to variable names for readability.

You can extend mapping by subclassing the corresponding `Mapping` class and adding the desired variables.

## Actions

The default `ActionsMapping` is as follows:

| Name                    | Value |
|-------------------------|-------|
| move_left               | 0     |
| move_right              | 1     |
| move_down               | 2     |
| rotate_clockwise        | 3     |
| rotate_counterclockwise | 4     |
| hard_drop               | 5     |
| swap                    | 6     |
| no-operation            | 7     |

## Rewards

The default `RewardsMapping` is as follows:

| Name                    | Value |
|-------------------------|-------|
| alife                   | 0.001 |
| clear_line              | 1     |
| game_over               | -2   |
