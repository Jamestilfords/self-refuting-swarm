"""Stream truth ignores canaries and non-ok proposals, and it is stable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skeptic.analysis.config import AnalysisConfig
from skeptic.analysis.sources import InMemorySource
from skeptic.analysis.truth import stream_truth_set


def test_canaries_and_failed_proposals_are_excluded():
    cell = "synth/bert"

    def pack(shift):
        ident = [np.zeros(8), np.zeros(8)]
        oos = [np.full(8, shift), np.full(8, shift)]
        return ident, oos

    source = InMemorySource("test", {
        (cell, "energy_ce"): pack(0.0),
        (cell, "good"): pack(3.0),
        (cell, "canary_sha"): pack(3.0),
        (cell, "broken"): pack(3.0),
        (cell, "null"): pack(0.0),
    })
    ledger = pd.DataFrame([
        {"sha": "good", "cell": cell, "status": "ok", "canary": False, "family": "score", "stream": "templated"},
        {"sha": "canary_sha", "cell": cell, "status": "ok", "canary": True, "family": "canary", "stream": "canary"},
        {"sha": "broken", "cell": cell, "status": "error", "canary": False, "family": "score", "stream": "llm"},
        {"sha": "null", "cell": cell, "status": "ok", "canary": False, "family": "score", "stream": "llm"},
    ])
    config = AnalysisConfig(
        cells=(cell,),
        test_B=80,
        test_seed=0,
        h1_cells=(cell,),
        h3_cells=(cell,),
        h4_small=(cell,),
        h4_large=(cell,),
        h2_min_cells=1,
    )
    first, _details = stream_truth_set(cell, ledger, source=source, config=config)
    second, _again = stream_truth_set(cell, ledger, source=source, config=config)
    assert first == second
    assert "good" in first
    assert "canary_sha" not in first
    assert "broken" not in first
    assert "null" not in first
