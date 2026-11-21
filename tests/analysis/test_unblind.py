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
from skeptic.analysis.decisions import load_decisions
from skeptic.analysis.report import _sources_from_unblind, build_report, main as report_main, verify_archive
from skeptic.analysis.report_tables import write_all
from skeptic.analysis.sources import GuardError, OpenNpzSource, SealedNpzSource, file_sha256
from skeptic.analysis.unblind import open_sealed
from skeptic.critic.er1 import softmax_ece
from skeptic.rules.common import REFERENCE_BASELINES, TEST_N_IN, VAL_N_IN
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


def _write_val_npz(repo: Path, cell: str, sha: str, *, n_in=None, finite=True):
    from skeptic.score_artifacts import CELLS

    n_in = VAL_N_IN[cell] if n_in is None else n_in
    scores = np.arange(5 * (n_in + 10), dtype=np.float32).reshape(5, n_in + 10)
    if not finite:
        scores = scores.astype(np.float64)
        scores[0, 0] = np.nan
    folder = repo / "runs" / "scores" / CELLS[cell]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sha}.npz"
    np.savez(path, scores=scores, n_in=n_in)
    return path


def _pin_score_manifest(repo: Path, paths: list[Path], references: dict) -> None:
    lines = [f"{file_sha256(path)}  {path.relative_to(repo).as_posix()}\n" for path in paths]
    manifest = repo / "runs" / "score_manifest.sha256"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(lines), encoding="ascii")
    decision = {
        "references": references,
        "replay": {"score_manifest_sha256": file_sha256(manifest)},
    }
    (repo / "runs" / "decision_manifest.json").write_text(json.dumps(decision), encoding="utf-8")


def _open_val(repo: Path, cell: str, sha: str, **kwargs):
    path = _write_val_npz(repo, cell, sha, **kwargs)
    references = {cell: {"energy_ce": sha}}
    _pin_score_manifest(repo, [path], references)
    return OpenNpzSource(repo / "runs" / "scores", references, repo=repo), path


def test_open_npz_resolves_names_and_refuses_bad_files(tmp_path):
    assert VAL_N_IN == {
        "clinc150/bert": 3000,
        "clinc150/roberta": 3000,
        "banking77_oos/bert": 1506,
        "banking77_oos/roberta": 1506,
    }
    source, path = _open_val(tmp_path, CELL, SHA)
    n_in = VAL_N_IN[CELL]
    ident, oos = source.get(CELL, "energy_ce")
    assert len(ident) == 5 and ident[0].dtype == np.float64
    assert ident[0].shape == (n_in,) and oos[0].shape == (10,)
    assert ident[0][0] == 0 and oos[0][0] == n_in
    assert source.arm_of(CELL, "energy_ce") == "ce"
    path.unlink()
    cached, _oos = source.get(CELL, SHA)
    assert cached[0][0] == ident[0][0]

    fresh, changed = _open_val(tmp_path / "mismatch", CELL, SHA)
    changed.write_bytes(changed.read_bytes() + b"\x00")
    with pytest.raises(GuardError, match="score manifest"):
        fresh.get(CELL, "energy_ce")
    missing = OpenNpzSource(
        tmp_path / "mismatch" / "runs" / "scores",
        {CELL: {"energy_ce": "c" * 12}},
        repo=tmp_path / "mismatch",
    )
    with pytest.raises(GuardError, match="missing"):
        missing.get(CELL, "c" * 12)

    short = _open_val(tmp_path / "short", CELL, "d" * 12, n_in=10)[0]
    with pytest.raises(GuardError, match="n_in"):
        short.get(CELL, "d" * 12)
    poisoned = _open_val(tmp_path / "nan", CELL, "e" * 12, finite=False)[0]
    with pytest.raises(GuardError, match="finite"):
        poisoned.get(CELL, "e" * 12)


def test_plain_model_ece_matches_the_validation_definition():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(40, 5))
    labels = rng.integers(0, 5, size=40)
    assert plain_model_ece(logits, labels, "ce") == pytest.approx(float(softmax_ece(logits, labels)), abs=1e-9)
    edl = plain_model_ece(logits, labels, "edl")
    assert np.isfinite(edl)


def _manifest(tmp_path: Path) -> None:
    from skeptic.score_artifacts import CELLS

    manifest = {
        "proposals": {cell: ["d" * 12] for cell in CELLS},
        "references": {cell: dict(REFERENCE_BASELINES) for cell in CELLS},
    }
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "decision_manifest.json").write_text(json.dumps(manifest))


def test_unblind_refuses_a_missing_file_and_records_a_complete_set(tmp_path, monkeypatch):
    _install(monkeypatch)
    _manifest(tmp_path)

    def missing(_commit, _digest, expected):
        assert "plain_ece.json" in expected
        return {"files": {}, "missing": ["plain_ece.json"]}

    with pytest.raises(FileNotFoundError, match="missing"):
        open_sealed(tmp_path, hasher=missing, require_registered=False)
    assert not (tmp_path / "runs" / "unblind.json").exists()
    assert not (tmp_path / "runs" / "guard.log").exists()

    def complete(_commit, _digest, expected):
        return {"files": {relative: "ab" * 32 for relative in expected}, "missing": []}

    record = open_sealed(tmp_path, hasher=complete, require_registered=False)
    assert record["counts"][CELL] == 1 + len(set(REFERENCE_BASELINES.values()))
    assert "plain_ece.json" in record["files"]
    assert (tmp_path / "runs" / "unblind.json").is_file()
    guard = (tmp_path / "runs" / "guard.log").read_text(encoding="utf-8")
    assert guard.count('purpose="unblind"') == 1


def test_hash_sealed_rejects_bad_paths_and_matches_file_bytes(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    from modal_jobs import unblind_remote

    _write_npz(tmp_path, CELL, SHA)
    (tmp_path / "plain_ece.json").write_bytes(b"{}\n")
    relative = f"scores/clinc150_bert/{SHA}.npz"
    events = []
    monkeypatch.setattr(unblind_remote, "SEALED_MOUNT", str(tmp_path))
    monkeypatch.setattr(unblind_remote, "sealed_volume", SimpleNamespace(reload=lambda: events.append("reload")))

    def verify(commit, digest):
        events.append("verify")
        return UnsealPermit(digest, commit)

    monkeypatch.setattr(unblind_remote, "verify_remote_freeze", verify)
    with pytest.warns(UserWarning, match="executing locally"):
        result = unblind_remote.hash_sealed.local(PERMIT.commit, PERMIT.digest, [relative, "plain_ece.json"])
    assert events == ["reload", "verify"]
    raw = (tmp_path / relative).read_bytes()
    assert result["missing"] == []
    assert result["files"][relative] == hashlib.sha256(raw).hexdigest()
    from skeptic.store import GuardError as StoreGuardError

    for bad in ("../x", "/tmp/x", "scores/clinc150_bert/../../x.npz"):
        with pytest.raises(StoreGuardError):
            unblind_remote.hash_sealed_files(tmp_path, PERMIT.commit, PERMIT.digest, [bad])
    link = tmp_path / "scores" / "clinc150_bert" / f"{'c' * 12}.npz"
    link.symlink_to(tmp_path / relative)
    with pytest.raises(StoreGuardError, match="symlink"):
        unblind_remote.hash_sealed_files(
            tmp_path, PERMIT.commit, PERMIT.digest, [f"scores/clinc150_bert/{'c' * 12}.npz"],
        )
    _write_npz(tmp_path, CELL, "e" * 12, commit="c" * 40)
    mismatched = unblind_remote.hash_sealed_files(
        tmp_path, PERMIT.commit, PERMIT.digest, [f"scores/clinc150_bert/{'e' * 12}.npz"],
    )
    assert mismatched == {"files": {}, "missing": [f"scores/clinc150_bert/{'e' * 12}.npz"]}


def test_hash_sealed_files_accepts_a_symlinked_root(tmp_path):
    from modal_jobs import unblind_remote

    real = tmp_path / "volume"
    real.mkdir()
    mount = tmp_path / "sealed"
    mount.symlink_to(real, target_is_directory=True)
    _write_npz(mount, CELL, SHA)
    (mount / "plain_ece.json").write_bytes(b"{}\n")
    relative = f"scores/clinc150_bert/{SHA}.npz"
    result = unblind_remote.hash_sealed_files(
        mount, PERMIT.commit, PERMIT.digest, [relative, "plain_ece.json"],
    )
    assert result["missing"] == []
    assert result["files"][relative] == file_sha256(real / relative)


def test_hash_sealed_files_rejects_a_link_below_the_root(tmp_path):
    from modal_jobs import unblind_remote
    from skeptic.store import GuardError as StoreGuardError

    real = tmp_path / "volume"
    real.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = real / "scores" / "clinc150_bert"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(outside, target_is_directory=True)
    sha = "c" * 12
    _write_npz(real, CELL, sha)
    with pytest.raises(StoreGuardError, match="symlink"):
        unblind_remote.hash_sealed_files(
            real, PERMIT.commit, PERMIT.digest, [f"scores/clinc150_bert/{sha}.npz"],
        )


def _hash_mount(monkeypatch, root: Path):
    from types import SimpleNamespace

    from modal_jobs import unblind_remote

    monkeypatch.setattr(unblind_remote, "SEALED_MOUNT", str(root))
    monkeypatch.setattr(unblind_remote, "sealed_volume", SimpleNamespace(reload=lambda: None))
    return unblind_remote


def test_authorized_hash_sealed_never_downloads(tmp_path, monkeypatch):
    from modal_jobs.authorization import freeze_receipt

    unblind_remote = _hash_mount(monkeypatch, tmp_path)
    _write_npz(tmp_path, CELL, SHA)
    (tmp_path / "plain_ece.json").write_bytes(b"{}\n")
    relative = f"scores/clinc150_bert/{SHA}.npz"
    expected = [relative, "plain_ece.json"]
    monkeypatch.setattr(unblind_remote, "verify_remote_freeze", lambda *_args, **_kwargs: pytest.fail("remote"))
    monkeypatch.setattr("skeptic.store._download", lambda *_args, **_kwargs: pytest.fail("download"))
    authorization = {"freeze": freeze_receipt(PERMIT, "unblind")}
    with pytest.warns(UserWarning, match="executing locally"):
        result = unblind_remote.hash_sealed.local(PERMIT.commit, PERMIT.digest, expected, authorization)
    assert result == unblind_remote.hash_sealed_files(tmp_path, PERMIT.commit, PERMIT.digest, expected)


@pytest.mark.parametrize("purpose,commit,digest", [
    ("unblind", "c" * 40, PERMIT.digest),
    ("unblind", PERMIT.commit, "d" * 64),
    ("plain_ece", PERMIT.commit, PERMIT.digest),
])
def test_wrong_unblind_receipt_refuses_before_any_file_is_opened(tmp_path, monkeypatch, purpose, commit, digest):
    from modal_jobs.authorization import freeze_receipt
    from skeptic.store import GuardError as StoreGuardError

    unblind_remote = _hash_mount(monkeypatch, tmp_path)
    _write_npz(tmp_path, CELL, SHA)
    monkeypatch.setattr(unblind_remote, "hash_sealed_files", lambda *_args, **_kwargs: pytest.fail("opened"))
    authorization = {"freeze": freeze_receipt(UnsealPermit(digest, commit), purpose)}
    with pytest.warns(UserWarning, match="executing locally"):
        with pytest.raises(StoreGuardError, match="Invalid run freeze authorization"):
            unblind_remote.hash_sealed.local(
                PERMIT.commit, PERMIT.digest, [f"scores/clinc150_bert/{SHA}.npz"], authorization,
            )


def test_standalone_hash_sealed_keeps_the_full_check(tmp_path, monkeypatch):
    unblind_remote = _hash_mount(monkeypatch, tmp_path)
    _write_npz(tmp_path, CELL, SHA)
    (tmp_path / "plain_ece.json").write_bytes(b"{}\n")
    calls = []

    def verify(commit, digest):
        calls.append((commit, digest))
        return UnsealPermit(digest, commit)

    monkeypatch.setattr(unblind_remote, "verify_remote_freeze", verify)
    expected = [f"scores/clinc150_bert/{SHA}.npz", "plain_ece.json"]
    with pytest.warns(UserWarning, match="executing locally"):
        unblind_remote.hash_sealed.local(PERMIT.commit, PERMIT.digest, expected, None)
    assert calls == [(PERMIT.commit, PERMIT.digest)]


def _archive(tmp_path: Path) -> tuple[Path, dict]:
    sealed = tmp_path / "sealed"
    path = _write_npz(sealed, CELL, SHA)
    plain = sealed / "plain_ece.json"
    plain.write_text("{}\n", encoding="utf-8")
    relative = f"scores/clinc150_bert/{SHA}.npz"
    record = {"files": {relative: file_sha256(path), "plain_ece.json": file_sha256(plain)}}
    return sealed, record


def test_report_refuses_a_modified_missing_or_extra_file(tmp_path):
    sealed, record = _archive(tmp_path)
    verify_archive(sealed, record)
    relative = next(iter(record["files"]))
    target = sealed / relative
    original = target.read_bytes()
    target.write_bytes(original + b"\x00")
    with pytest.raises(ValueError, match="hash"):
        verify_archive(sealed, record)
    target.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_archive(sealed, record)
    target.write_bytes(original)
    extra = _write_npz(sealed, CELL, "d" * 12)
    with pytest.raises(ValueError, match="extra"):
        verify_archive(sealed, record)
    extra.unlink()
    verify_archive(sealed, record)


def test_report_runs_on_a_matching_archive(tmp_path, monkeypatch, capsys):
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
    sealed, record = _archive(tmp_path)
    payload = {
        "schema_version": 1,
        "permit": {"commit": PERMIT.commit, "digest": PERMIT.digest},
        "counts": {CELL: 1},
        "files": record["files"],
    }
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "unblind.json").write_text(json.dumps(payload), encoding="utf-8")
    written = build_report(
        tmp_path, decisions=decisions, ledger=stream.ledger,
        test_source=stream.test, val_source=stream.val, config=config,
        out_dir=tmp_path / "tables", decisions_sha256="a" * 64,
    )
    exploratory = (tmp_path / "tables" / "exploratory" / "exp1_per_ordering.csv").read_text(encoding="utf-8")
    assert ",100," in exploratory
    assert "hypotheses.csv" in written
    monkeypatch.setattr(
        "skeptic.analysis.report._sources_from_unblind",
        lambda *_args, **_kwargs: (stream.ledger, stream.test, stream.val),
    )
    monkeypatch.setattr("skeptic.analysis.report.load_decisions", lambda *_args, **_kwargs: decisions)
    monkeypatch.setattr("skeptic.analysis.report.build_report", lambda *_args, **_kwargs: written)
    assert report_main(["--repo", str(tmp_path), "--sealed", str(sealed), "--out", str(tmp_path / "out")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "reported"
    target = sealed / next(iter(record["files"]))
    target.write_bytes(target.read_bytes() + b"\x00")
    assert report_main(["--repo", str(tmp_path), "--sealed", str(sealed), "--out", str(tmp_path / "out")]) == 1
    assert "hash" in capsys.readouterr().err


def test_report_runs_on_a_matching_archive_with_validation_npz(tmp_path, monkeypatch, capsys):
    _install(monkeypatch)
    cell = CELL
    candidate, other = "a" * 12, "b" * 12
    energy = REFERENCE_BASELINES["energy_ce"]
    monkeypatch.setitem(VAL_N_IN, cell, 30)
    monkeypatch.setitem(TEST_N_IN, cell, 30)
    sealed = tmp_path / "sealed"
    sealed_paths = {sha: _write_npz(sealed, cell, sha, n_in=30) for sha in (energy, candidate, other)}
    plain = sealed / "plain_ece.json"
    plain.write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "permit": {"commit": PERMIT.commit, "digest": PERMIT.digest},
        "counts": {cell: 3},
        "files": {
            **{f"scores/clinc150_bert/{sha}.npz": file_sha256(path) for sha, path in sealed_paths.items()},
            "plain_ece.json": file_sha256(plain),
        },
    }
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "unblind.json").write_text(json.dumps(payload), encoding="utf-8")
    val_paths = [_write_val_npz(tmp_path, cell, sha, n_in=30) for sha in (energy, candidate, other)]
    _pin_score_manifest(tmp_path, val_paths, {cell: dict(REFERENCE_BASELINES)})
    ledger_rows = [
        {"sha": candidate, "cell": cell, "status": "ok", "stream": "templated"},
        {"sha": other, "cell": cell, "status": "ok", "stream": "llm"},
    ]
    (tmp_path / "runs" / "ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger_rows), encoding="utf-8",
    )
    decisions = [
        {"cell": cell, "sha": candidate, "idx": 0, "ordering": 0, "rule": "R0",
         "incumbent": "energy_ce", "verdict": "ACCEPT", "accepted": True},
        {"cell": cell, "sha": other, "idx": 1, "ordering": 0, "rule": "R0",
         "incumbent": candidate, "verdict": "REJECT", "accepted": False},
    ]
    folder = tmp_path / "runs" / "decisions" / "R0"
    folder.mkdir(parents=True)
    (folder / "ord_000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in decisions), encoding="utf-8")

    ledger, test, val = _sources_from_unblind(tmp_path, sealed)
    assert isinstance(val, OpenNpzSource)
    loaded = load_decisions(tmp_path / "runs" / "decisions", ledger)
    build_report(
        tmp_path, decisions=loaded, ledger=ledger, test_source=test, val_source=val,
        config=_small_config(cell), out_dir=tmp_path / "tables",
        decisions_sha256=PERMIT.digest, workers=1,
    )
    winners = (tmp_path / "tables" / "exp1_winners_curse.csv").read_text(encoding="utf-8")
    assert "mean_delta_val" in winners
    body = [line for line in winners.splitlines() if line and not line.startswith("#")]
    assert len(body) > 1 and body[1].split(",")[-2] not in {"", "nan"}

    def spy(repo, **kwargs):
        assert isinstance(kwargs["val_source"], OpenNpzSource)
        ident, oos = kwargs["val_source"].get(cell, "energy_ce")
        assert ident[0].shape == (30,) and oos[0].shape == (10,)
        return {}

    monkeypatch.setattr("skeptic.analysis.report.build_report", spy)
    assert report_main(["--repo", str(tmp_path), "--sealed", str(sealed), "--out", str(tmp_path / "out"), "--workers", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "reported"
    target = sealed / f"scores/clinc150_bert/{energy}.npz"
    target.write_bytes(target.read_bytes() + b"\x00")
    assert report_main(["--repo", str(tmp_path), "--sealed", str(sealed), "--out", str(tmp_path / "out"), "--workers", "1"]) == 1
    assert "hash" in capsys.readouterr().err


def _small_config(cell: str) -> AnalysisConfig:
    return AnalysisConfig(
        cells=(cell,), test_B=20, test_seed=0, hyp_B=20, hyp_seed=0,
        registered_orderings=100, h1_cells=(cell,), h3_cells=(cell,),
        h4_small=(cell,), h4_large=(cell,), h2_min_cells=1,
    )


def _report_inputs(tmp_path: Path):
    stream = make_stream(seed=0, n_true=2, n_null=2, n_seeds=2, sizes={"val": (30, 15), "test": (30, 15)})
    decisions = replay(stream.val, stream.shas, stream.cell, n_orderings=2, seed=0)
    extra = decisions.loc[decisions["ordering"] == 0].copy()
    extra["ordering"] = 100
    decisions = pd.concat([decisions, extra], ignore_index=True)
    (tmp_path / "runs").mkdir(exist_ok=True)
    record = tmp_path / "runs" / "unblind.json"
    if not record.is_file():
        record.write_text("{}\n", encoding="utf-8")
    return stream, decisions, _small_config(stream.cell)


def _published_tables(root: Path) -> dict[str, bytes]:
    tables = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix == ".csv" or path.name == "SUMMARY.md"):
            tables[path.relative_to(root).as_posix()] = path.read_bytes()
    return tables


def _count_bootstrap(monkeypatch):
    import skeptic.analysis.truth as truth

    calls = {"n": 0}
    real = truth.paired_bootstrap_fast

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(truth, "paired_bootstrap_fast", wrapped)
    return calls


def test_report_caches_every_pair_and_skips_them_on_rerun(tmp_path, monkeypatch):
    stream, decisions, config = _report_inputs(tmp_path)
    cache = tmp_path / "runs" / "truth"
    first = tmp_path / "first"
    second = tmp_path / "second"
    calls = _count_bootstrap(monkeypatch)
    build_report(
        tmp_path, decisions=decisions, ledger=stream.ledger,
        test_source=stream.test, val_source=stream.val, config=config,
        out_dir=first, decisions_sha256="a" * 64, cache_dir=cache, workers=1,
    )
    computed = calls["n"]
    assert computed > 0
    calls["n"] = 0
    build_report(
        tmp_path, decisions=decisions, ledger=stream.ledger,
        test_source=stream.test, val_source=stream.val, config=config,
        out_dir=second, decisions_sha256="a" * 64, cache_dir=cache, workers=1,
    )
    assert calls["n"] == 0
    assert _published_tables(first) == _published_tables(second)


def test_report_one_and_two_workers_write_the_same_tables(tmp_path):
    stream, decisions, config = _report_inputs(tmp_path)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    common = dict(
        decisions=decisions, ledger=stream.ledger, test_source=stream.test,
        val_source=stream.val, config=config, decisions_sha256="a" * 64,
    )
    build_report(tmp_path, out_dir=serial, cache_dir=tmp_path / "cache-1", workers=1, **common)
    build_report(tmp_path, out_dir=parallel, cache_dir=tmp_path / "cache-2", workers=2, **common)
    assert _published_tables(serial) == _published_tables(parallel)


def test_report_reads_published_decision_format(tmp_path):
    from skeptic.analysis.decisions import load_decisions

    stream, decisions, config = _report_inputs(tmp_path)
    energy = "b1e4ba2e46d5"
    for (rule, ordering), group in decisions.groupby(["rule", "ordering"], sort=True):
        folder = tmp_path / "decisions" / str(rule)
        folder.mkdir(parents=True, exist_ok=True)
        lines = []
        for record in group.sort_values("idx").to_dict(orient="records"):
            record.pop("accepted", None)
            for key in ("incumbent", "comparator"):
                if record.get(key) == "energy_ce":
                    record[key] = energy
            record["detail"] = {"G1": [1, 2, 3]}
            lines.append(json.dumps(record))
        (folder / f"ord_{int(ordering):03d}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    loaded = load_decisions(tmp_path / "decisions", ledger=stream.ledger)
    assert "detail" not in loaded.columns
    original = decisions.sort_values(["rule", "ordering", "idx"])
    loaded = loaded.sort_values(["rule", "ordering", "idx"])
    assert [
        (str(row.rule), int(row.ordering), str(row.sha), bool(row.accepted), str(row.incumbent))
        for row in loaded.itertuples(index=False)
    ] == [
        (str(row.rule), int(row.ordering), str(row.sha), bool(row.accepted), str(row.incumbent))
        for row in original.itertuples(index=False)
    ]
    common = dict(
        ledger=stream.ledger, test_source=stream.test, val_source=stream.val,
        config=config, decisions_sha256="a" * 64, workers=1,
    )
    direct = tmp_path / "direct"
    published = tmp_path / "published"
    build_report(
        tmp_path, decisions=decisions, out_dir=direct, cache_dir=tmp_path / "cache-direct", **common,
    )
    build_report(
        tmp_path, decisions=loaded, out_dir=published, cache_dir=tmp_path / "cache-published", **common,
    )
    assert _published_tables(direct) == _published_tables(published)


def test_report_resumes_from_cached_pairs(tmp_path, monkeypatch):
    import skeptic.analysis.truth as truth

    stream, decisions, config = _report_inputs(tmp_path)
    cache = tmp_path / "runs" / "truth"
    real = truth.paired_bootstrap_fast
    state = {"n": 0}

    def stop_after_two(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 3:
            raise RuntimeError("stop")
        return real(*args, **kwargs)

    monkeypatch.setattr(truth, "paired_bootstrap_fast", stop_after_two)
    with pytest.raises(RuntimeError, match="stop"):
        build_report(
            tmp_path, decisions=decisions, ledger=stream.ledger,
            test_source=stream.test, val_source=stream.val, config=config,
            out_dir=tmp_path / "partial", decisions_sha256="a" * 64,
            cache_dir=cache, workers=1,
        )
    assert len(list(cache.rglob("*.json"))) == 2
    monkeypatch.undo()
    calls = _count_bootstrap(monkeypatch)
    resumed = tmp_path / "resumed"
    clean_cache = tmp_path / "clean-cache"
    clean = tmp_path / "clean"
    common = dict(
        decisions=decisions, ledger=stream.ledger, test_source=stream.test,
        val_source=stream.val, config=config, decisions_sha256="a" * 64, workers=1,
    )
    build_report(tmp_path, out_dir=resumed, cache_dir=cache, **common)
    resumed_calls = calls["n"]
    calls["n"] = 0
    build_report(tmp_path, out_dir=clean, cache_dir=clean_cache, **common)
    assert resumed_calls == calls["n"] - 2
    assert _published_tables(resumed) == _published_tables(clean)


def test_sealed_npz_source_pickles(tmp_path, monkeypatch):
    import pickle

    _install(monkeypatch)
    sealed = tmp_path / "runs" / "sealed"
    _write_npz(sealed, CELL, SHA)
    source = SealedNpzSource(sealed, PERMIT, {CELL: {"energy_ce": SHA}}, repo=tmp_path)
    original = source.get(CELL, "energy_ce")
    restored = pickle.loads(pickle.dumps(source))
    again = restored.get(CELL, "energy_ce")
    for left, right in zip(original, again):
        for left_row, right_row in zip(left, right):
            assert np.array_equal(left_row, right_row)


def test_sealed_source_rechecks_the_unblind_hash(tmp_path, monkeypatch):
    _install(monkeypatch)
    sealed = tmp_path / "runs" / "sealed"
    path = _write_npz(sealed, CELL, SHA)
    relative = f"scores/clinc150_bert/{SHA}.npz"
    source = SealedNpzSource(
        sealed, PERMIT, {CELL: {"energy_ce": SHA}}, repo=tmp_path,
        verified={relative: file_sha256(path)},
    )
    source.get(CELL, "energy_ce")
    path.write_bytes(path.read_bytes() + b"\x00")
    again = SealedNpzSource(
        sealed, PERMIT, {CELL: {"energy_ce": SHA}}, repo=tmp_path,
        verified={relative: file_sha256(path) + "00"},
    )
    with pytest.raises(GuardError, match="hash"):
        again.get(CELL, "energy_ce")


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
