"""A variant that accepts a superset of the critic's accepts slips the extra false ones."""

from __future__ import annotations

import pandas as pd
import pytest

from skeptic.analysis.ablation import gate_ablation
from skeptic.analysis.config import AnalysisConfig


def test_slipped_and_false_count_match_the_hand_count():
    cell = "synth/bert"
    rows = [
        ("R2", 0, "a", "energy_ce", False),
        ("R2", 0, "b", "energy_ce", True),
        ("R2-noG3", 0, "a", "energy_ce", False),
        ("R2-noG3", 0, "b", "energy_ce", True),
        ("R2-noG3", 0, "c", "energy_ce", True),
        ("R2", 1, "a", "energy_ce", False),
        ("R2-noG3", 1, "a", "energy_ce", False),
        ("R2-noG3", 1, "d", "a", True),
    ]
    changes = pd.DataFrame(rows, columns=["rule", "ordering", "sha", "incumbent", "is_false"])
    changes["cell"] = cell
    config = AnalysisConfig(
        cells=(cell,),
        hyp_B=30,
        h1_cells=(cell,),
        h3_cells=(cell,),
        h4_small=(cell,),
        h4_large=(cell,),
        h2_min_cells=1,
        gate_variants=("R2-noG3",),
    )
    table = gate_ablation(
        changes,
        proposals=["a", "b", "c", "d"],
        orderings=[0, 1],
        config=config,
    )
    row = table.iloc[0]
    assert row["d_n_false_vs_R2"] == pytest.approx(1.0)
    assert row["slipped_mean"] == pytest.approx(1.0)
    assert row["n_false_mean"] == pytest.approx(1.5)
