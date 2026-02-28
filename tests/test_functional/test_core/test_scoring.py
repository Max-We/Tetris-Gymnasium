"""Tests for score formula."""

import pytest

from tetris_gymnasium.functional.core import EnvConfig, score

CONFIG = EnvConfig(width=10, height=20, padding=4, queue_size=7)


class TestScore:
    @pytest.mark.parametrize(
        "rows_cleared,expected",
        [(0, 0), (1, 100), (2, 300), (3, 500), (4, 800)],
    )
    def test_score_formula(self, rows_cleared, expected):
        result = score(CONFIG, rows_cleared)
        assert int(result) == expected
