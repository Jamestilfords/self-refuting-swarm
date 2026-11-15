"""The fast bootstrap equals ER-1's, and the bridge feeds ledger + score files to the merged critic."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ER1 = Path(__file__).resolve().parents[1] / "third_party" / "er1" / "src"
sys.path.insert(0, str(ER1))
from bert_daedl.posthoc.stats import paired_bootstrap as er1_bootstrap  # noqa: E402

from skeptic.fastboot import paired_bootstrap  # noqa: E402

N_IN, N_OUT, SEEDS = 300, 80, 5


def _halves(xs):
    return [x[:N_IN] for x in xs], [x[N_IN:] for x in xs]


def test_fast_bootstrap_equals_er1():
    rng = np.random.default_rng(1)
    a = [np.round(rng.normal(size=N_IN + N_OUT) + 1.5 * (np.arange(N_IN + N_OUT) >= N_IN), 1) for _ in range(SEEDS)]   # ties
    b = [x + rng.normal(size=x.size) * 0.4 for x in a]
    ref = er1_bootstrap(*_halves(a), *_halves(b), B=200, seed=7)
    fast = paired_bootstrap(*_halves(a), *_halves(b), B=200, seed=7)
    assert set(fast) == set(ref)
    for key in ref:
        assert fast[key] == pytest.approx(ref[key], abs=1e-12)


def test_bridge_feeds_the_critic(tmp_path):
    pytest.importorskip("torch")                                         # the merged critic imports ER-1, which needs torch
    from skeptic.critic import LordPlusPlus, verdict
    from skeptic.critic.baselines import REGISTERED_NAMES
    from skeptic.critic_inputs import load_cell

    rng = np.random.default_rng(3)
    truth = np.r_[np.zeros(N_IN), np.ones(N_OUT)]
    noise = lambda sd: rng.normal(0, sd, (SEEDS, truth.size))
    sc = {name: truth + noise(0.8 + 0.05 * i) for i, name in enumerate(REGISTERED_NAMES)}
    sc["inert"] = 2 * sc["energy_ce"] + 1
    sc["renamed"] = np.exp(sc["msp_ce"])
    sc["noise"] = noise(1.0)
    sc["strong"] = truth + noise(0.25)
    sha = {name: f"{i:012x}" for i, name in enumerate(sc)}
    folder = tmp_path / "scores" / "clinc150_bert"
    folder.mkdir(parents=True)
    rows = []
    for name, s in sc.items():
        np.savez_compressed(folder / f"{sha[name]}.npz", scores=s.astype(np.float32), n_in=N_IN)
        rows.append({"sha": sha[name], "cell": "clinc150/bert", "status": "ok", "arm": "edl" if name.endswith("_edl") else "ce",
                     "stream": "templated", "family": "other", "parents": [],
                     "val": {"ece": None, "ece_plain": [0.02] * SEEDS}})
    rows.append({"sha": "f" * 12, "cell": "clinc150/bert", "status": "io", "arm": "ce", "stream": "canary", "val": None})
    rows.append({"sha": "e" * 12, "cell": "clinc150/bert", "status": "timeout", "arm": "ce", "stream": "llm", "val": None})
    (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "baselines.json").write_text(json.dumps([{"name": n.replace("_ce", "@ce").replace("_edl", "@edl"), "sha": sha[n]}
                                                         for n in REGISTERED_NAMES]))
    cell = load_cell("clinc150/bert", tmp_path / "scores", tmp_path / "ledger.jsonl", tmp_path / "baselines.json")
    assert cell.malformed == {"e" * 12: "timeout"} and cell.incumbent.name == "energy_ce"
    judge = lambda name: verdict(cell.candidates[name], cell.incumbent, cell.baselines, LordPlusPlus(), cell.cell).verdict.value
    assert judge(sha["inert"]) == "INERT"
    assert judge(sha["renamed"]) == "SCORE-MATCHED"
    assert judge(sha["noise"]) == "NO EFFECT"
    assert judge("f" * 12) == "BUG"
    assert judge(sha["strong"]) == "ACCEPT"
