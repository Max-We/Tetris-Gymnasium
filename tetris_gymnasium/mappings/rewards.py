"""This module contains the mapping for the rewards that the agent can receive."""
from dataclasses import dataclass


@dataclass
class RewardsMapping:
    """Mapping for the rewards that the agent can receive.

    The mapping can be extended to include additional rewards.
    """

    alife: float = 1
    clear_line: float = 1
    game_over: float = 0
    invalid_action: float = -0.1
