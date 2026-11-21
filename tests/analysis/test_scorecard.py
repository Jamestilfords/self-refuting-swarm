"""Registered scorecard pairs grade strict, lenient, or miss."""

from pathlib import Path

from skeptic.analysis.report import _hypothesis_line, _scorecard_shas
from skeptic.analysis.scorecard import PAIRS, grade, match_counts

ROOT = Path(__file__).resolve().parents[2]


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


def test_unwhitened_pair_uses_the_templated_proposal_hash():
    published = _scorecard_shas(ROOT)
    assert published[2] == ("71faf9e6269e", "c0fdb8397000")
    assert PAIRS[1].incumbent == "d_port_edl_unwhitened"
    assert PAIRS[1].candidate == "d_port_edl"


def test_summary_states_the_registered_support_rule():
    h2 = _hypothesis_line(
        {
            "hypothesis": "H2",
            "estimate": "",
            "ci_low": "",
            "ci_high": "",
            "p_holm": "1.000000",
            "supported_by_rule": "false",
            "n_orderings": "100",
            "n_excluded_orderings": "400",
        },
        critic_accepts=0,
        n_cells=4,
    )
    assert h2 == (
        "H2: not evaluable: R2 accepted 0 changes in all 400 cell-orderings (all excluded); "
        "supported if 95% CI > 0 in at least 3 of the 4 cells and Holm-adjusted p ≤ 0.05"
    )
    h3 = _hypothesis_line(
        {
            "hypothesis": "H3",
            "estimate": "0.000000",
            "ci_low": "0.000000",
            "ci_high": "0.000000",
            "p_holm": "0.002000",
            "supported_by_rule": "false",
            "n_orderings": "100",
            "n_excluded_orderings": "0",
        },
        critic_accepts=0,
        n_cells=4,
    )
    assert "supported=false" in h3
    assert "reject=" not in h3
    assert "point estimate is at least 0.5" in h3


def test_lenient_count_includes_strict_matches():
    grades = {}
    for pair in PAIRS:
        grades[pair.number] = "strict" if pair.number <= 4 else "lenient" if pair.number <= 7 else "miss"
    counts = match_counts(grades)
    assert counts == {"n_pairs": 8, "strict_matches": 4, "lenient_matches": 7}
