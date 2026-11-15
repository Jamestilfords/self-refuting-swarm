"""L4 and L5: memory, Proposer and Scout, tested with a scripted LLM and a fake Exa client (no network)."""
import json
import shutil
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from skeptic import api
from skeptic.memory import LocalMemory, format_for_prompt
from skeptic.streams.proposer import Proposer, extract_code, smoke_ctx
from skeptic.streams.scout import Scout, top_rejection


def proposal(body: str, *, source: str = "", memory_used=(), family: str = "other") -> str:
    head = (f'FAMILY = "{family}"\nARM = "ce"\nRATIONALE = "test idea"\nSOURCE = "{source}"\n'
            f'PARENTS = []\nMEMORY_USED = {json.dumps(list(memory_used))}\n\n\ndef score(ctx):\n')
    return "```python\n" + head + textwrap.indent(textwrap.dedent(body).strip(), "    ") + "\n```"


KNN = "return ctx.h.knn(ctx.train.pooled, ctx.eval.pooled, k=5)"
DENSITY = "g = ctx.h.gda(ctx.train.feats, ctx.train.labels)\nreturn g.min_mahalanobis_sq(ctx.eval.feats)"
ENERGY_COPY = "return 3.0 * ctx.h.energy(ctx.eval.logits)"
BROKEN = "import os\nreturn ctx.h.energy(ctx.eval.logits)"


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, system, user):
        self.prompts.append(user)
        return self.replies.pop(0), {"input_tokens": 100, "output_tokens": 50}


@pytest.fixture(scope="module")
def ctx():
    return smoke_ctx()


@pytest.fixture()
def work(tmp_path):
    out = tmp_path / "proposals"
    out.mkdir()
    shutil.copy("proposals/_template.py", out / "_template.py")
    return out, LocalMemory(tmp_path / "memory.jsonl")


# ---------------------------------------------------------------- memory
def test_memory_ids_persistence_and_sealed_guard(tmp_path):
    m = LocalMemory(tmp_path / "m.jsonl")
    a = m.write("source", "scout", {"url": "https://example.org/a", "title": "A", "method": "kNN distance"})
    b = m.write("verdict", "critic", {"sha": "abc", "verdict": "INERT", "summary": "no pair changed order"})
    assert (a, b) == ("m_0001", "m_0002") and LocalMemory(tmp_path / "m.jsonl").ids() == {a, b}
    assert "m_0001 source" in format_for_prompt(m.read()) and top_rejection(m) == "INERT"
    with pytest.raises(ValueError):
        m.write("lesson", "analyst", {"text": "auroc on oos_test was 0.98"})
    with pytest.raises(ValueError):
        m.write("lesson", "nobody", {"text": "x"})


# ---------------------------------------------------------------- proposer
def test_accepts_a_valid_new_proposal(work, ctx):
    out, mem = work
    p = Proposer(mem, ScriptedLLM([proposal(KNN, family="knn")]), ctx, out, model="fake")
    r = p.propose_one(0)
    assert r["status"] == "ok" and (out / f"{r['sha']}.py").exists() and r["attempts"] == 1
    card = mem.read(["proposal"])[0]
    assert card["author"] == "proposer" and card["content"]["sha"] == r["sha"]
    p.save()
    assert json.loads((out / "llm.json").read_text())["proposals"][0]["stream"] == "llm"


def test_repairs_after_a_rejection(work, ctx):
    out, mem = work
    llm = ScriptedLLM([proposal(BROKEN), proposal(KNN)])
    r = Proposer(mem, llm, ctx, out).propose_one(0)
    assert r["status"] == "ok" and r["attempts"] == 2
    assert "rejected" in llm.prompts[1] and "import" in llm.prompts[1]        # the reason went back to the LLM


def test_rejects_copies_of_known_scores(work, ctx):
    out, mem = work
    p = Proposer(mem, ScriptedLLM([proposal(ENERGY_COPY), proposal(ENERGY_COPY + "  # again")]), ctx, out)
    r = p.propose_one(0)
    assert r["status"] == "rejected" and r["reason"] == "duplicate"
    assert list(out.glob("[0-9a-f]*.py")) == []                               # nothing left on disk


def test_second_proposal_must_differ_from_the_first(work, ctx):
    out, mem = work
    same = "return 2.0 * ctx.h.knn(ctx.train.pooled, ctx.eval.pooled, k=5) + 1.0"
    p = Proposer(mem, ScriptedLLM([proposal(KNN), proposal(same), proposal(DENSITY)]), ctx, out)
    assert p.propose_one(0)["status"] == "ok"
    r = p.propose_one(1)                                                      # first try is a rescaled copy, repaired
    assert r["status"] == "ok" and r["attempts"] == 2 and len(p.rows) == 2


def test_citations_must_exist_in_memory(work, ctx):
    out, mem = work
    fake_id = proposal(KNN, memory_used=["m_0999"])
    fake_url = proposal(KNN, source="https://arxiv.org/abs/0000.00000")
    p = Proposer(mem, ScriptedLLM([fake_id, fake_url]), ctx, out)
    r = p.propose_one(0)
    assert r["status"] == "rejected" and r["reason"] == "source"              # the last failure is reported


def test_resume_reloads_earlier_proposals(work, ctx):
    out, mem = work
    p = Proposer(mem, ScriptedLLM([proposal(KNN)]), ctx, out)
    sha = p.propose_one(0)["sha"]
    p.save()
    again = Proposer(mem, ScriptedLLM([proposal(KNN + "  # same idea")]), ctx, out, max_attempts=1)
    assert sha in again.known and again.propose_one(1)["reason"] == "duplicate"


# ---------------------------------------------------------------- scout, and the two together
class FakeExa:
    def __init__(self):
        self.queries = []

    def search_and_contents(self, query, num_results, text):
        self.queries.append(query)
        return SimpleNamespace(results=[
            SimpleNamespace(url="https://example.org/knn", title="Deep nearest neighbours", published_date="2022-04-01",
                            text="We score by the distance to the k-th nearest training embedding."),
            SimpleNamespace(url="https://example.org/blog", title="A blog", published_date=None, text="Nothing useful."),
        ])


def test_scout_writes_source_cards_and_proposer_cites_them(work, ctx):
    out, mem = work
    exa = FakeExa()
    llm = ScriptedLLM(["Score each item by its distance to the k-th nearest l2-normalized training embedding.", "NONE"])
    scout = Scout(mem, llm, exa)
    ids = scout.search()
    assert ids == ["m_0001"] and exa.queries and mem.read(["source"])[0]["content"]["url"] == "https://example.org/knn"
    assert scout.search() == []                                               # same urls are not written twice

    llm2 = ScriptedLLM([proposal(KNN, source="https://example.org/knn", memory_used=["m_0001"], family="knn")])
    p = Proposer(mem, llm2, ctx, out)
    r = p.propose_one(0)
    assert r["status"] == "ok" and r["memory_used"] == ["m_0001"] and r["source"] == "https://example.org/knn"
    assert "https://example.org/knn" in llm2.prompts[0]                       # the proposer saw the scout's card
    assert set(mem.stats()) == {"scout", "proposer"}                          # two agents wrote; one read the other's


def test_loop_survives_llm_errors_and_scout_failures(work, ctx, capsys):
    out, mem = work

    class Flaky(ScriptedLLM):
        def __call__(self, system, user):
            if not self.replies:
                raise RuntimeError("rate limited")
            return super().__call__(system, user)

    class BadScout:
        def search(self):
            raise RuntimeError("exa down")

    p = Proposer(mem, Flaky([proposal(KNN)]), ctx, out)
    p.run(2, scout=BadScout())
    assert len(p.rows) == 1 and p.rejected[0]["reason"] == "llm_error"
    assert "search failed" in capsys.readouterr().out


def test_extract_code():
    assert extract_code("text\n```python\nx = 1\n```\nmore") == "x = 1\n"
    assert extract_code("x = 2") == "x = 2\n"


# ---------------------------------------------------------------- past.dev backend (HTTP calls replaced)
def test_pastdev_memory_ingests_every_write_and_recalls_with_ids(tmp_path, work, ctx):
    from skeptic.memory import PastDevMemory
    out, _ = work
    calls = []

    class Fake(PastDevMemory):
        def _call(self, method, path, body=None):
            calls.append((method, path, body))
            if path == "/recall":
                return {"results": [{"rank": 1, "content": "[m_0001] Source Deep nearest neighbours describes: kNN distance",
                                     "sources": [{"id": "m_0001"}]}]}
            return {"ingestionId": "ing_1", "status": "queued"}

    mem = Fake(tmp_path / "m.jsonl", api_key="k")
    sid = mem.write("source", "scout", {"url": "https://example.org/knn", "title": "Deep nearest neighbours", "method": "kNN distance"})
    assert calls[0][1] == "/ingest" and calls[0][2]["id"] == sid and "[m_0001]" in calls[0][2]["content"]
    assert "audience" not in calls[0][2]                                       # project-visible: every agent can read it
    assert mem.recall("knn")[0]["ids"] == ["m_0001"]

    llm = ScriptedLLM([proposal(KNN, source="https://example.org/knn", memory_used=["m_0001"], family="knn")])
    r = Proposer(mem, llm, ctx, out).propose_one(0)
    assert r["status"] == "ok" and "Recalled from shared memory (past.dev)" in llm.prompts[0]
    assert mem.ingested == 2 and mem.recalls == 2 and mem.errors == 0          # source card + proposal card


def test_pastdev_outage_does_not_stop_the_loop(tmp_path, capsys):
    import urllib.error
    from skeptic.memory import PastDevMemory

    class Down(PastDevMemory):
        def _call(self, method, path, body=None):
            raise urllib.error.URLError("no route")

    mem = Down(tmp_path / "m.jsonl", api_key="k")
    assert mem.write("lesson", "analyst", {"text": "mul couplings miscalibrate"}) == "m_0001"
    assert mem.recall("anything") == [] and mem.errors == 2 and mem.get("m_0001") is not None
    assert capsys.readouterr().out.count("[past.dev]") == 1                    # warned once, not per call


def test_cache_ctx_reads_the_open_volume_format(tmp_path):
    from skeptic.streams.proposer import cache_ctx
    rng = np.random.default_rng(0)
    arrays = {"train__labels": rng.integers(0, 4, 50)}
    for split, n in (("train", 50), ("val", 20), ("oos_val", 7)):
        arrays.update({f"{split}__logits": rng.normal(size=(n, 4)), f"{split}__feats": rng.normal(size=(n, 8)),
                       f"{split}__pooled": rng.normal(size=(n, 12))})
    np.savez_compressed(tmp_path / "cache.npz", **arrays)
    c = cache_ctx(tmp_path)
    assert len(c.train) == 50 and len(c.eval) == 27 and c.eval.labels is None and c.eval.pooled.shape == (27, 12)
