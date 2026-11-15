"""Runner and sealer core on small fake caches shaped like the real open and sealed stores."""
import json

import numpy as np
import pytest

from skeptic import api, runner
from skeptic.store import GuardError
from skeptic.streams import baselines, canaries
from skeptic.streams.common import render, write

GROUP, K = "clinc150_bert", 5


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    """<open>/<group>/<arm>_seed<k>/cache.npz and <sealed>/.../{cache,pooled}.npz for both arms, five seeds."""
    root = tmp_path_factory.mktemp("stores")
    open_root, sealed_root = root / "open", root / "sealed"
    sizes = {"train": 300, "val": 80, "test": 120, "oos_val": 20, "oos_test": 40}
    for arm in ("ce", "edl"):
        for seed in runner.SEEDS:
            rng = np.random.default_rng(100 * seed + (arm == "edl"))
            cache, pooled = {}, {}
            for split, n in sizes.items():
                oos = split.startswith("oos")
                y = rng.integers(0, K, n)
                shift = 1.5 if oos else 0.0
                cache[f"{split}__logits"] = (np.eye(K)[y] * (2 if oos else 6) + rng.normal(size=(n, K))).astype(np.float32)
                cache[f"{split}__feats"] = (np.eye(K, 12)[y] * 4 + rng.normal(size=(n, 12)) + shift).astype(np.float32)
                cache[f"{split}__labels"] = np.full(n, -1) if oos else y
                pooled[f"{split}__pooled"] = (np.eye(K, 20)[y] * 4 + rng.normal(size=(n, 20)) + shift).astype(np.float32)
            d = sealed_root / GROUP / f"{arm}_seed{seed}"
            d.mkdir(parents=True)
            np.savez_compressed(d / "cache.npz", **cache)
            np.savez_compressed(d / "pooled.npz", **pooled)
            o = open_root / GROUP / f"{arm}_seed{seed}"
            o.mkdir(parents=True)
            keep = {f"{s}__{f}" for s in ("train", "val", "oos_val") for f in ("logits", "feats", "pooled")}
            merged = {**cache, **pooled}
            np.savez_compressed(o / "cache.npz", train__labels=cache["train__labels"], **{k: merged[k] for k in keep})
    return open_root, sealed_root


@pytest.fixture(scope="module")
def props(tmp_path_factory):
    d = tmp_path_factory.mktemp("proposals")
    shas = {r["name"]: r["sha"] for r in canaries.write_canaries(d)}
    shas.update({r["name"]: r["sha"] for r in baselines.write_baselines(d) if r["name"] in ("energy@ce", "msp@edl")})
    shas["coupling"] = write(d, render(family="fusion", arm="edl", rationale="density coupling", body="""
        tr, ev, h = ctx.train, ctx.eval, ctx.h
        g = h.gda(tr.feats, tr.labels)
        p_hat = h.normalizer("ecdf", g.log_density(tr.feats))(g.log_density(ev.feats))
        return h.couple_mul(ev.logits, p_hat)
    """))
    shas["crash"] = write(d, render(family="other", arm="ce", rationale="breaks", body="return ctx.eval.labels + 1"))
    return d, shas


def test_validation_rows_and_score_files(stores, props):
    open_root, _ = stores
    d, shas = props
    rows = {r["sha"]: r for r in runner.score_validation(d, list(shas.values()), open_root, GROUP)}
    ok = rows[shas["energy@ce"]]
    assert ok["status"] == "ok" and ok["arm"] == "ce" and len(ok["val"]["auroc"]) == 5
    assert all(a > 0.8 for a in ok["val"]["auroc"])                     # OOS logits are flatter, so energy separates them
    z = np.load(open_root / "scores" / GROUP / f"{shas['energy@ce']}.npz")
    assert z["scores"].shape == (5, 100) and int(z["n_in"]) == 80
    assert rows[shas["coupling"]]["has_probs"] and not ok["has_probs"]
    assert rows[shas["test_leak"]]["status"] == "io"                    # the leak canary is stopped before it runs
    assert rows[shas["crash"]]["status"] == "runtime_error"
    assert rows[shas["gaussian_noise"]]["status"] == "ok"
    assert abs(np.mean(rows[shas["gaussian_noise"]]["val"]["auroc"]) - 0.5) < 0.15


def test_canaries_behave_as_planted(stores, props):
    open_root, _ = stores
    d, shas = props
    runner.score_validation(d, list(shas.values()), open_root, GROUP)
    load = lambda name: np.load(open_root / "scores" / GROUP / f"{shas[name]}.npz")["scores"]
    energy, inert = load("energy@ce"), load("inert_rescale")
    assert all((np.argsort(a) == np.argsort(b)).all() for a, b in zip(energy, inert))   # same ranking as the incumbent


def test_resume_reuses_stored_scores(stores, props):
    open_root, _ = stores
    d, shas = props
    first = runner.score_validation(d, [shas["energy@ce"]], open_root, GROUP)[0]
    again = runner.score_validation(d, [shas["energy@ce"]], open_root, GROUP)[0]
    assert again["val"] == first["val"] and "wall_s" not in again


def test_legacy_sealer_is_disabled_before_reading_test(stores, props):
    open_root, sealed_root = stores
    d, shas = props
    good = [shas["energy@ce"], shas["coupling"], shas["msp@edl"]]
    runner.score_validation(d, good, open_root, GROUP)
    with pytest.raises(GuardError, match="Legacy test scorer is disabled"):
        runner.score_test(d, good, open_root, sealed_root, GROUP)
    assert not (sealed_root / "scores").exists()


def test_calibration_runs_no_proposal_and_reads_only_validation_labels(stores, props, tmp_path):
    open_root, sealed_root = stores
    d, shas = props
    good = [shas["energy@ce"], shas["coupling"], shas["msp@edl"]]
    runner.score_validation(d, good, open_root, GROUP)
    labels_only = tmp_path / "sealed"                                    # a sealed store holding nothing but validation labels
    for arm in ("ce", "edl"):
        for seed in runner.SEEDS:
            (labels_only / GROUP / f"{arm}_seed{seed}").mkdir(parents=True)
            y = np.load(sealed_root / GROUP / f"{arm}_seed{seed}" / "cache.npz")["val__labels"]
            np.savez_compressed(labels_only / GROUP / f"{arm}_seed{seed}" / "cache.npz", val__labels=y)
    items = [(shas["energy@ce"], "ce"), (shas["coupling"], "edl"), (shas["msp@edl"], "edl")]
    rows = {r["sha"]: r for r in runner.validation_calibration(open_root, labels_only, GROUP, items)}
    assert rows[shas["energy@ce"]]["val_ece"] is None                    # scores only: G5 does not apply
    assert len(rows[shas["coupling"]]["val_ece"]) == 5                   # probs returned: calibration measured
    assert all(len(r["val_ece_plain"]) == 5 and 0 <= min(r["val_ece_plain"]) for r in rows.values())
    z = np.load(sealed_root / GROUP / "edl_seed0" / "cache.npz")
    o = np.load(open_root / "scores" / GROUP / f"{shas['coupling']}.npz")
    assert rows[shas["coupling"]]["val_ece"][0] == pytest.approx(runner.ece(o["conf"][0, :80], o["pred"][0, :80] == z["val__labels"]), abs=1e-6)


def test_metrics():
    assert runner.auroc([0, 1], [2, 3]) == 1.0 and runner.auroc([2, 3], [0, 1]) == 0.0
    assert runner.auroc([1, 1], [1, 1]) == 0.5
    assert runner.ece([0.9, 0.9, 0.9, 0.9], [1, 1, 1, 0]) == pytest.approx(0.15)


def test_plan_and_ledger_merge(stores, props, tmp_path):
    open_root, sealed_root = stores
    d, shas = props
    found = runner.discover(d)
    assert found[shas["test_leak"]] == {"stream": "canary", "arm": "ce"}
    assert found[shas["energy@ce"]]["stream"] == "baseline" and shas["coupling"] not in found   # not in a manifest
    tasks = runner.build_tasks(found)
    assert {t["cell"] for t in tasks} == set(runner.CELLS) and all(len(t["shas"]) <= runner.BATCH for t in tasks)
    assert all(found[s]["arm"] == t["arm"] for t in tasks for s in t["shas"])

    mine = [t for t in tasks if t["group"] == GROUP]
    val = {(r["sha"], t["cell"]): r for t in mine for r in runner.score_validation(d, t["shas"], open_root, GROUP)}
    good = lambda t: [s for s in t["shas"] if val[(s, t["cell"])]["status"] == "ok"]
    cal = {(r["sha"], t["cell"]): r for t in mine
           for r in runner.validation_calibration(open_root, sealed_root, GROUP, [(s, t["arm"]) for s in good(t)])}
    ledger = runner.merge_ledger(found, tasks, val, cal, {})
    unsealed = runner.merge_ledger(found, tasks, val, cal, {})
    assert next(r for r in unsealed if r["sha"] == shas["energy@ce"] and r["cell"] == "clinc150/bert")["sealed"] == "not_run"
    assert len(ledger) == len(found) * 4
    row = next(r for r in ledger if r["sha"] == shas["energy@ce"] and r["cell"] == "clinc150/bert")
    assert row["status"] == "ok" and row["sealed"] == "not_run" and row["stream"] == "baseline"
    assert set(row["val"]) == {"auroc", "fpr95", "ece", "ece_plain"} and len(row["val"]["ece_plain"]) == 5
    leak = next(r for r in ledger if r["sha"] == shas["test_leak"] and r["cell"] == "clinc150/bert")
    assert leak["status"] == "io" and leak["sealed"] == "skipped" and leak["val"] is None
    other = next(r for r in ledger if r["sha"] == shas["energy@ce"] and r["cell"] == "banking77_oos/roberta")
    assert other["status"] == "infra_error"                              # that cell's batches never reported back
    runner.write_ledger(ledger, tmp_path / "ledger.jsonl")
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == len(ledger)
    assert "canary" in runner.summarize(ledger)
