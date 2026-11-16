"""Sealed npz source, reference check, plain ECE, unblind, and report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modal_jobs.plain_ece import plain_model_ece
from skeptic.analysis.check_references import arrays_match, first_mismatch, spearman
from skeptic.analysis.config import AnalysisConfig
from skeptic.analysis.report import build_report
from skeptic.analysis.report_tables import write_all
from skeptic.analysis.sources import GuardError, SealedNpzSource
from skeptic.analysis.unblind import open_sealed
from skeptic.critic.er1 import softmax_ece
from skeptic.rules.common import REFERENCE_BASELINES, TEST_N_IN
from skeptic.store import UnsealPermit
from tests.fixtures.analysis.replay import replay
from tests.fixtures.analysis.synth_stream import make_stream

PERMIT = UnsealPermit("a" * 64, "b" * 40)
CELL = "clinc150/bert"
SHA = REFERENCE_BASELINES["energy_ce"]


def _install(monkeypatch, permit=PERMIT):
    import skeptic.store as store

    monkeypatch.setattr(store, "verify_published_freeze", lambda repo: permit)
    monkeypatch.setattr("skeptic.analysis.unblind.verify_published_freeze", lambda repo: permit)


def _write_npz(root: Path, cell: str, sha: str, *, n_in=None, scores=None, commit=PERMIT.commit,
               digest=PERMIT.digest, finite=True):
    from skeptic.score_artifacts import CELLS

    n_in = TEST_N_IN[cell] if n_in is None else n_in
    if scores is None:
        scores = np.arange(5 * (n_in + 10), dtype=np.float64).reshape(5, n_in + 10)
        if not finite:
            scores[0, 0] = np.nan
    folder = root / "scores" / CELLS[cell]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sha}.npz"
    np.savez(path, scores=scores, n_in=n_in, proposal_sha=sha,
             decision_commit=commit, decision_digest=digest)
    return path


def test_spearman_is_one_for_a_copy_and_fails_a_reversal():
    left = np.arange(6, dtype=np.float64)
    assert spearman(left, left + 3) == pytest.approx(1.0)
    assert spearman(left, left[::-1]) < 1
    rows = [left + index for index in range(5)]
    assert arrays_match(np.vstack(rows), rows)
    assert not arrays_match(np.vstack(rows), [row[::-1] for row in rows])


def test_check_references_reports_the_first_missing_file(tmp_path):
    problem = first_mismatch(tmp_path / "scores", tmp_path / "baselines")
    assert problem is not None and problem.startswith("missing")


def test_sealed_npz_resolves_names_and_refuses_bad_files(tmp_path, monkeypatch):
    _install(monkeypatch)
    sealed = tmp_path / "runs" / "sealed"
    _write_npz(sealed, CELL, SHA)
    source = SealedNpzSource(
        sealed, PERMIT, {CELL: {"energy_ce": SHA}}, repo=tmp_path,
    )
    ident, oos = source.get(CELL, "energy_ce")
    assert len(ident) == 5 and ident[0].shape == (TEST_N_IN[CELL],)
    assert oos[0].shape == (10,)
    assert source.probabilities(CELL, "energy_ce") is None
    assert source.arm_of(CELL, "energy_ce") == "ce"
    path = sealed / "scores" / "clinc150_bert" / f"{SHA}.npz"
    path.unlink()
    ident2, _oos2 = source.get(CELL, "energy_ce")
    assert ident2[0][0] == ident[0][0]
    log = (sealed / "guard.log").read_text(encoding="utf-8")
    assert log.count("energy_ce") == 2

    other = "c" * 12
    _write_npz(sealed, CELL, other, n_in=10)
    with pytest.raises(GuardError, match="n_in"):
        source.get(CELL, other)
    _write_npz(sealed, CELL, other, finite=False)
    with pytest.raises(GuardError, match="finite"):
        source.get(CELL, other)
    _write_npz(sealed, CELL, other, commit="c" * 40)
    with pytest.raises(GuardError, match="provenance"):
        source.get(CELL, other)
    _write_npz(sealed, CELL, other)
    link = sealed / "scores" / "clinc150_bert" / f"{other}.npz"
    raw = link.read_bytes()
    link.unlink()
    target = sealed / "elsewhere.npz"
    target.write_bytes(raw)
    link.symlink_to(target)
    with pytest.raises(GuardError, match="symlink"):
        source.get(CELL, other)


def test_plain_model_ece_matches_the_validation_definition():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(40, 5))
    labels = rng.integers(0, 5, size=40)
    assert plain_model_ece(logits, labels, "ce") == pytest.approx(float(softmax_ece(logits, labels)), abs=1e-9)
    edl = plain_model_ece(logits, labels, "edl")
    assert np.isfinite(edl)


def test_unblind_refuses_a_missing_file_and_records_a_complete_set(tmp_path, monkeypatch):
    from skeptic.score_artifacts import CELLS

    _install(monkeypatch)
    sealed = tmp_path / "sealed"
    manifest = {
        "proposals": {cell: ["d" * 12] for cell in CELLS},
        "references": {cell: dict(REFERENCE_BASELINES) for cell in CELLS},
    }
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "decision_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="missing"):
        open_sealed(tmp_path, sealed, require_registered=False)
    for cell in CELLS:
        _write_npz(sealed, cell, "d" * 12)
        for sha in set(REFERENCE_BASELINES.values()):
            _write_npz(sealed, cell, sha)
    (sealed / "plain_ece.json").write_text("{}\n")
    record = open_sealed(tmp_path, sealed, require_registered=False)
    assert record["counts"][CELL] == 1 + len(set(REFERENCE_BASELINES.values()))
    assert (tmp_path / "runs" / "unblind.json").is_file()
    assert 'purpose="unblind"' in (tmp_path / "runs" / "guard.log").read_text(encoding="utf-8")


def test_report_needs_unblind_and_keeps_later_orderings_exploratory(tmp_path):
    with pytest.raises(FileNotFoundError, match="unblind"):
        build_report(
            tmp_path, decisions=pd.DataFrame(), ledger=pd.DataFrame(),
            test_source=None, val_source=None,
        )
    stream = make_stream(seed=0, n_true=2, n_null=2, n_seeds=2, sizes={"val": (30, 15), "test": (30, 15)})
    decisions = replay(stream.val, stream.shas, stream.cell, n_orderings=2, seed=0)
    extra = decisions.loc[decisions["ordering"] == 0].copy()
    extra["ordering"] = 100
    decisions = pd.concat([decisions, extra], ignore_index=True)
    cell = stream.cell
    config = AnalysisConfig(
        cells=(cell,), test_B=20, test_seed=0, hyp_B=20, hyp_seed=0,
        registered_orderings=100, h1_cells=(cell,), h3_cells=(cell,),
        h4_small=(cell,), h4_large=(cell,), h2_min_cells=1,
    )
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "unblind.json").write_text("{}\n")
    written = build_report(
        tmp_path, decisions=decisions, ledger=stream.ledger,
        test_source=stream.test, val_source=stream.val, config=config,
        out_dir=tmp_path / "tables", decisions_sha256="a" * 64,
    )
    confirmatory = (tmp_path / "tables" / "exp1_per_ordering.csv").read_text(encoding="utf-8")
    exploratory = (tmp_path / "tables" / "exploratory" / "exp1_per_ordering.csv").read_text(encoding="utf-8")
    assert ",100," not in confirmatory
    assert ",100," in exploratory
    assert "H1" in (tmp_path / "tables" / "SUMMARY.md").read_text(encoding="utf-8")
    assert "hypotheses.csv" in written
