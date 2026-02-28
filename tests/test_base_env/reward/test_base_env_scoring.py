import pytest

from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_gymnasium.mappings.rewards import RewardsMapping


@pytest.mark.parametrize(
    "lines,expected_score",
    [
        (1, 1**2 * 10),  # 10
        (2, 2**2 * 10),  # 40
        (3, 3**2 * 10),  # 90
        (4, 4**2 * 10),  # 160
    ],
)
def test_score_formula(tetris_env, lines, expected_score):
    """Test that score formula (n^2) * width is correct for different line counts."""
    tetris_env.reset(seed=42)
    score = tetris_env.unwrapped.score(lines)
    assert score == expected_score


def test_alife_reward_on_successful_placement(tetris_env):
    """Test that +1 survival reward is given on successful piece placement."""
    tetris_env.reset(seed=42)

    _, reward, terminated, _, _ = tetris_env.step(ActionsMapping.hard_drop)

    assert not terminated
    # Reward should include alife reward (1) plus any line clear score
    assert reward >= RewardsMapping.alife


def test_game_over_reward_is_zero(tetris_env):
    """Test that the reward on game over is 0."""
    tetris_env.reset(seed=42)

    # Fill the board to cause game over
    tetris_env.unwrapped.board[
        0 : tetris_env.unwrapped.height,
        tetris_env.unwrapped.padding : -tetris_env.unwrapped.padding,
    ] = 2

    _, reward, terminated, _, _ = tetris_env.step(ActionsMapping.hard_drop)

    assert terminated
    assert reward == RewardsMapping.game_over
