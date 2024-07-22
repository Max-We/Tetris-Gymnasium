"""This module contains mappings for actions that the agent can take."""
from dataclasses import dataclass


@dataclass
class ActionsMapping:
    """The actions that the agent can take.

    The mapping can be extended to include additional rewards.
    """

    move_left: int = 0
    move_right: int = 1
    move_down: int = 2
    rotate_clockwise: int = 3
    rotate_counterclockwise: int = 4
    hard_drop: int = 5
    swap: int = 6
    no_op: int = 7
