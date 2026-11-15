"""Registered scorecard pairs grade strict, lenient, or miss."""

from skeptic.analysis.scorecard import PAIRS, grade, match_counts


def test_eight_pairs_are_registered_in_order():
    assert [pair.number for pair in PAIRS] == list(range(1, 9))
    assert PAIRS[0].candidate == "vacuity_edl"
    assert PAIRS[5].cell == "banking77_oos/bert"
    assert PAIRS[5].candidate == "d_port_edl"


def test_underpowered_is_lenient_on_the_power_pairs():
    for number in (5, 6, 7):
        pair = PAIRS[number - 1]
        assert grade("ACCEPT", pair) == "strict"
        assert grade("UNDERPOWERED", pair) == "lenient"
        assert grade("NO EFFECT", pair) == "miss"


def test_pair_three_accepts_miscalibration_as_lenient():
    pair = PAIRS[2]
    assert grade("NO EFFECT", pair) == "strict"
    assert grade("MISCALIBRATES", pair) == "lenient"


def test_lenient_count_includes_strict_matches():
    grades = {}
    for pair in PAIRS:
        grades[pair.number] = "strict" if pair.number <= 4 else "lenient" if pair.number <= 7 else "miss"
    counts = match_counts(grades)
    assert counts == {"n_pairs": 8, "strict_matches": 4, "lenient_matches": 7}
