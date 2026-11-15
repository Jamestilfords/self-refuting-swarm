"""Experiment 2: the closed loop and the Analyst, with a scripted LLM and local scoring on synthetic arrays."""
import json
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

from skeptic import api
from skeptic.analysis import exp2
from skeptic.memory import LocalMemory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "er1" / "src"))
pytest.importorskip("torch")                                             # the critic imports ER-1, which needs torch

from skeptic.critic import BaselineSet, ScoreSet                          # noqa: E402
from skeptic.critic.baselines import G2_NAMES, REGISTERED_NAMES          # noqa: E402
from skeptic.critic_inputs import Cell                                    # noqa: E402
from skeptic.streams.proposer import smoke_ctx                            # noqa: E402

N_IN, SEEDS = 300, 5


def proposal(body: str, family: str = "other", memory_used=()) -> str:
    head = (f'FAMILY = "{family}"\nARM = "ce"\nRATIONALE = "test idea"\nSOURCE = ""\n'
            f'PARENTS = []\nMEMORY_USED = {json.dumps(list(memory_used))}\n\n\ndef score(ctx):\n')
    return "```python\n" + head + textwrap.indent(textwrap.dedent(body).strip(), "    ") + "\n```"


KNN = proposal("return ctx.h.knn(ctx.train.pooled, ctx.eval.pooled, k=5)", "knn")
KNN_FEATS = proposal("return ctx.h.knn(ctx.train.feats, ctx.eval.feats, k=7)", "knn")
DENSITY = proposal("g = ctx.h.gda(ctx.train.feats, ctx.train.labels)\nreturn g.min_mahalanobis_sq(ctx.eval.feats)", "density")
ENERGY_COPY = proposal("return 3.0 * ctx.h.energy(ctx.eval.logits)", "logit")
RELATIVE = proposal("g = ctx.h.gda(ctx.train.pooled, ctx.train.labels)\nreturn g.relative_mahalanobis_sq(ctx.eval.pooled)", "density")


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, system, user):
        self.prompts.append(user)
        return self.replies.pop(0), {"input_tokens": 100, "output_tokens": 50}


@pytest.fixture(scope="module")
def world():
    """Five synthetic seeds, a local scorer, and a cell whose baselines are computed on the same items."""
    exp2.use_fast_bootstrap()
    ctxs = [smoke_ctx(seed) for seed in range(SEEDS)]
    calls = []

    def score(text):
        calls.append(api.proposal_sha(text))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / f"{api.proposal_sha(text)}.py"
            path.write_text(text)
            p = api.load_proposal(path)
        return {"scores": np.stack([api.validate(p, c)["scores"] for c in ctxs]), "n_in": N_IN, "ece": None}

    rng = np.random.default_rng(0)
    truth = np.r_[np.zeros(N_IN), np.ones(len(ctxs[0].eval) - N_IN)]

    def from_rows(name, rows):
        return ScoreSet(name=name, arm="edl" if name.endswith("_edl") else "ce",
                        val_id=tuple(r[:N_IN] for r in rows), val_oos=tuple(r[N_IN:] for r in rows))

    base = {"energy_ce": from_rows("energy_ce", [c.h.energy(c.eval.logits) for c in ctxs]),
            "msp_ce": from_rows("msp_ce", [c.h.msp(c.eval.logits) for c in ctxs])}
    for name in REGISTERED_NAMES:
        if name not in base:
            base[name] = from_rows(name, [0.3 * truth + rng.normal(size=truth.size) for _ in ctxs])
    baselines = BaselineSet(cell="banking77_oos/bert", scores=base, g2_names=G2_NAMES,
                            plain_ece={"ce": (0.02,) * SEEDS, "edl": (0.02,) * SEEDS})
    return Cell("banking77_oos/bert", {}, baselines, base["energy_ce"], {}, {}), score, calls


def make(arm, world, tmp_path, replies, analyst=None, **kw):
    cell, score, _ = world
    run_dir = tmp_path / arm
    (run_dir / "proposals").mkdir(parents=True, exist_ok=True)
    shutil.copy("proposals/_template.py", run_dir / "proposals" / "_template.py")
    llm = replies if isinstance(replies, ScriptedLLM) else ScriptedLLM(replies)
    loop = exp2.Loop(arm, cell, run_dir, llm=llm, score=score, model="scripted", memory=LocalMemory(run_dir / "memory.jsonl"),
                     analyst_llm=analyst, log=lambda *_: None, **kw)
    return loop, llm


def test_arm_b_closes_the_loop(world, tmp_path):
    analyst = ScriptedLLM(["Density scores keep being judged no better than the baseline; try the encoder output."])
    loop, llm = make("B", world, tmp_path, [KNN, ENERGY_COPY, ENERGY_COPY, DENSITY, RELATIVE], analyst, analyst_every=2)
    rows = loop.run(4)
    assert [r["status"] for r in rows] == ["judged", "not_scored", "judged", "judged"]
    assert rows[1]["reason"] == "duplicate"                               # a rescaled baseline never reaches the critic
    for r in (rows[0], rows[2], rows[3]):
        assert r["verdict"] in {"ACCEPT", "INERT", "SCORE-MATCHED", "NO EFFECT", "UNDERPOWERED", "MISCALIBRATES", "BUG"}
        assert r["rule"] == "critic" and "critic" in r and r["memory_id"]
    stats = loop.memory.stats()
    assert stats["critic"]["verdict"] == 3 and stats["proposer"]["proposal"] == 3 and stats["analyst"]["lesson"] == 1
    assert f"{rows[0]['sha']} -> {rows[0]['verdict']}" in llm.prompts[-1]              # the next prompt carries the verdicts
    assert "lesson: Density scores keep" in llm.prompts[-1]                # ... and the Analyst's lesson
    assert len(analyst.prompts) == 1 and rows[0]["sha"] in analyst.prompts[0]
    saved = [json.loads(line) for line in (tmp_path / "B" / "decisions.jsonl").read_text().splitlines()]
    assert saved == rows and json.loads((tmp_path / "B" / "state.json").read_text())["slots"] == 4


def test_arm_c_sees_nothing_and_arm_a_sees_only_its_history(world, tmp_path):
    loop_c, llm_c = make("C", world, tmp_path, [KNN, DENSITY])
    rows_c = loop_c.run(2)
    assert rows_c[0]["sha"] not in llm_c.prompts[1] and "m_0001" not in llm_c.prompts[1]
    assert loop_c.memory.stats()["critic"]["verdict"] == 2                # still recorded, just not shown
    assert loop_c.analyst is None

    loop_a, llm_a = make("A", world, tmp_path, [KNN, DENSITY], ScriptedLLM([]))
    rows_a = loop_a.run(2)
    assert {r["verdict"] for r in rows_a} <= {"KEPT", "DISCARDED"} and rows_a[0]["rule"] == "ratchet"
    assert rows_a[0]["sha"] in llm_a.prompts[1]                           # its own kept/discarded history
    assert loop_a.analyst is None and loop_a.lord.t == 0                  # no LORD++ test under the ratchet
    assert rows_a[0]["accepted"] == (rows_a[0]["delta_inc"] > 0)
    if rows_a[0]["accepted"]:
        assert rows_a[1]["incumbent"] == rows_a[0]["sha"]


def test_resume_continues_without_rescoring(world, tmp_path):
    _, _, calls = world
    loop, _ = make("C", world, tmp_path, [KNN_FEATS, DENSITY])
    first = loop.run(2)
    before = len(calls)
    again, llm = make("C", world, tmp_path, [RELATIVE])
    assert len(calls) == before                                           # earlier proposals reload from the score files
    rows = again.run(3)
    assert rows[:2] == first and rows[2]["idx"] == 2 and len(calls) == before + 1
    assert again.lord.to_dict() == json.loads((tmp_path / "C" / "state.json").read_text())["lord"]
    with pytest.raises(ValueError, match="another arm"):
        exp2.Loop("B", world[0], tmp_path / "C", llm=llm, score=world[1], model="scripted",
                  memory=LocalMemory(tmp_path / "C" / "memory.jsonl"))


def test_wasted_share_and_report(tmp_path, capsys):
    def judged(idx, family, verdict, accepted=False):
        return {"idx": idx, "arm": "B", "cell": "banking77_oos/bert", "status": "judged", "family": family,
                "verdict": verdict, "accepted": accepted, "val_auroc": 0.9, "sha": f"{idx:012x}"}
    rows = [judged(0, "density", "NO EFFECT"), judged(1, "density", "NO EFFECT"), judged(2, "knn", "ACCEPT", True),
            {"idx": 3, "arm": "B", "cell": "banking77_oos/bert", "status": "not_scored", "reason": "duplicate",
             "verdict": None, "accepted": False},
            judged(4, "density", "UNDERPOWERED"), judged(5, "fusion", "NO EFFECT"), judged(6, "density", "NO EFFECT"),
            judged(7, "density", "UNDERPOWERED")]
    assert exp2.wasted_flags([r for r in rows if r["status"] == "judged"]) == [False, True, False, False, False, True, True]
    run = tmp_path / "banking77_bert" / "B0"
    run.mkdir(parents=True)
    (run / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = exp2.report(tmp_path)[0]
    assert (summary["slots"], summary["judged"], summary["accepted"], summary["wasted"]) == (8, 7, 1, 3)
    assert summary["wasted_first_half"] == pytest.approx(1 / 3, abs=1e-4) and summary["wasted_second_half"] == 0.5
    assert "arm B: mean change in wasted share +0.167" in capsys.readouterr().out
