"""Boundary, cache, and the refusal to score validation arrays as test truth."""

from __future__ import annotations

import numpy as np
import pytest

from skeptic.analysis.config import AnalysisConfig
from skeptic.analysis.sources import InMemorySource
from skeptic.analysis.truth import TruthCache, true_improvement


def _config():
    cell = "cell"
    return AnalysisConfig(
        cells=(cell,),
        test_B=30,
        test_seed=1,
        h1_cells=(cell,),
        h3_cells=(cell,),
        h4_small=(cell,),
        h4_large=(cell,),
        h2_min_cells=1,
    )


def _source(kind="test"):
    zeros = np.zeros(6)
    return InMemorySource(kind, {
        ("cell", "cand"): ([zeros, zeros], [zeros, zeros]),
        ("cell", "inc"): ([zeros, zeros], [zeros, zeros]),
    })


def test_interval_touching_zero_is_not_a_true_improvement(monkeypatch):
    def fake(*_args, **_kwargs):
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "se": 0.0, "p": 1.0}

    monkeypatch.setattr("skeptic.analysis.truth.paired_bootstrap_fast", fake)
    result = true_improvement("cand", "inc", "cell", source=_source(), config=_config())
    assert result.ci95[0] == 0.0
    assert result.true_primary is False

    def positive(*_args, **_kwargs):
        return {"delta": 1e-6, "ci_low": 1e-6, "ci_high": 1e-6, "se": 0.0, "p": 0.05}

    monkeypatch.setattr("skeptic.analysis.truth.paired_bootstrap_fast", positive)
    assert true_improvement("cand", "inc", "cell", source=_source(), config=_config()).true_primary is True


def test_validation_source_is_rejected():
    with pytest.raises(ValueError, match="test source"):
        true_improvement("cand", "inc", "cell", source=_source("val"), config=_config())


def test_cache_hit_skips_the_bootstrap(tmp_path, monkeypatch):
    calls = {"n": 0}
    real = __import__("skeptic.analysis.bootstrap", fromlist=["paired_bootstrap_fast"]).paired_bootstrap_fast

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("skeptic.analysis.truth.paired_bootstrap_fast", wrapped)
    cache = TruthCache(tmp_path / "truth")
    config = _config()
    source = _source()
    first = true_improvement("cand", "inc", "cell", source=source, config=config, cache=cache)
    assert calls["n"] == 1
    second = true_improvement("cand", "inc", "cell", source=source, config=config, cache=cache)
    assert calls["n"] == 1
    assert second == first
