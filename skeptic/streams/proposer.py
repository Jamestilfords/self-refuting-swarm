"""L5: the Proposer. An LLM writes one score function per idea; the validator decides whether it enters.

    python -m skeptic.streams.proposer --n 5                 # small first run
    python -m skeptic.streams.proposer --n 100 --cache DIR   # check against a real validation cache

For each proposal the loop: reads memory, asks the LLM for one file, runs the static and dynamic
checks of slide 8, rejects near-copies of earlier scores (Spearman >= 0.999), writes the file under
its hash, and writes a proposal card back to memory. Every 10 proposals the Scout adds source cards
from an Exa search. Rejections go back to the LLM once for repair.

Guards on what the LLM claims: MEMORY_USED must list real memory ids, and SOURCE must be the url of a
source card the Scout wrote (or empty), so no citation can be invented.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from skeptic import api
from skeptic.memory import format_for_prompt, get_memory
from skeptic.streams.common import write_manifest

DEFAULT_MODEL = "gpt-5.6-terra"
DISTINCT = 0.999          # Spearman at or above this to an earlier score means "not distinct" (slide 14)
SCOUT_EVERY = 10          # † one Exa search per 10 proposals

API_DOC = """
ctx.train   .logits (N,K)  .feats (N,256) bottleneck  .pooled (N,768) encoder output  .labels (N,) ints
ctx.eval    .logits  .feats  .pooled   (no labels; you cannot tell which items are out-of-scope)
ctx.h       helpers, all returning numpy arrays, higher score = more out-of-scope:
  h.softmax(logits) -> (N,K)      h.energy(logits)        h.energy_clip(logits)     h.msp(logits)
  h.entropy(logits)               h.vacuity(logits)       h.alpha(logits) -> (N,K) Dirichlet concentrations
  h.l2(x) -> row-normalized x     h.pca(train_x, dim).transform(x) -> (N,dim)
  g = h.gda(train_x, labels, cov="tied", pca=None, whiten=True)    # Gaussian per class, shared covariance
      g.min_mahalanobis_sq(x)  g.class_mahalanobis_sq(x) -> (N,C)  g.neg_log_density(x)  g.log_density(x)
      g.relative_mahalanobis_sq(x)
  h.mahalanobis_pp(train_x, labels, x, pca=None)      h.relative_mahalanobis(train_x, labels, x, pca=None)
  p = h.partial(train_x, labels, top_k=128); p.top(x); p.bottom(x)   # leading / trailing eigen-directions
  h.knn(train_x, x, k=10, l2=True)                    # distance to the k-th nearest training item
  n = h.normalizer("minmax" | "qsigmoid" | "ecdf", train_log_density); p_hat = n(log_density)  # in [0,1]
  h.couple_mul(logits, p_hat) -> (scores, probs)      h.couple_add_log(logits, p_hat, lam) -> (scores, probs)
You may also import numpy, scipy and sklearn (not scipy.io, not sklearn.datasets).
""".strip()

SYSTEM = """
You are a research engineer proposing out-of-scope (OOS) detectors for an intent classifier.
You write ONE proposal per reply: a complete Python file in a single ```python block, nothing else.

The file must define these six constants and one function, exactly as in the example:
  FAMILY       one of "logit", "density", "knn", "fusion", "other"
  ARM          "ce" or "edl" (which fine-tuned model's cached arrays you want)
  RATIONALE    one or two sentences: why this should separate OOS from in-scope queries
  SOURCE       the url of a source card from memory if the idea came from it, else ""
  PARENTS      for a fusion of registered scores, a list drawn from "energy", "msp", "d_port", "d_full"; else []
  MEMORY_USED  the ids (like "m_0007") of the memory entries that shaped this proposal; [] if none
  def score(ctx): returns one float per item in ctx.eval, higher = more out-of-scope

Hard rules. A file that breaks one is rejected:
  - Fit everything on ctx.train. Each eval item's score must depend only on that item and on ctx.train.
    Never use statistics of ctx.eval (no mean, std, rank, min or max over the eval scores or features).
  - Deterministic. Any random direction must come from np.random.default_rng(<fixed int>).
  - Imports only from numpy, scipy, sklearn. No file or network access, no open, eval, exec, getattr.
  - Finite, non-constant output of shape (len(ctx.eval),). Under 60 seconds with 15,000 train items,
    150 classes and 768 dimensions: do not build an (items, classes, dims) array yourself.
  - Return (scores, probs) only if your method changes the predicted class probabilities.
  - Do not repeat an earlier proposal or a plain baseline (energy, msp, a single GDA density).
""".strip()

ANGLES = [
    "Fuse the two densities: one on the bottleneck features (good at near-OOS, held-out intents) and one on "
    "the encoder output (good at far-OOS, other domains). Additive fusion, weights fitted on train only.",
    "A residual-space score in the spirit of ViM: the norm of the feature component outside the principal "
    "subspace of the training features, alone or added to energy.",
    "A k-nearest-neighbour score: vary k, the feature space, l2 normalization, or average the k distances.",
    "Use the trailing (low-variance) eigen-directions of the within-class covariance, where the paper found the OOS signal.",
    "A class-conditional score: distance to the predicted class's mean rather than the nearest class, or the gap "
    "between the nearest and second-nearest class distances.",
    "A logit-shape score beyond energy and max-probability: top-two logit margin, logit norm, or a tempered energy.",
    "Cosine similarity to class prototypes on l2-normalized features, turned into an OOS score.",
    "Combine a logit score with a feature-space distance after putting both on a common scale using train statistics only.",
    "An evidential score for the EDL arm that is NOT a monotone function of energy (vacuity is), for example one that "
    "uses the shape of the Dirichlet concentrations.",
    "A score that agrees with energy on confident items but uses feature distance where energy is uncertain.",
    "Per-class variance: a diagonal or shrunk covariance per class instead of one tied covariance.",
    "Project features onto the classifier's most discriminative directions (e.g. class-mean subspace) and score the remainder.",
]


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def openai_llm(model: str = DEFAULT_MODEL):
    """Returns llm(system, user) -> (text, usage). Reads OPENAI_API_KEY from the environment."""
    from openai import OpenAI
    client = OpenAI()

    def llm(system: str, user: str):
        r = client.responses.create(model=model, instructions=system, input=user)
        u = getattr(r, "usage", None)
        return r.output_text, {"input_tokens": getattr(u, "input_tokens", 0) or 0,
                               "output_tokens": getattr(u, "output_tokens", 0) or 0}
    return llm


def smoke_ctx(seed: int = 0) -> api.Ctx:
    """Synthetic arrays with the real widths (256-d, 768-d). Only for checking that a proposal runs;
    distinctness on real data is re-checked from the ledger."""
    rng = np.random.default_rng(seed)
    k = 30

    def split(n, shift=0.0):
        y = rng.integers(0, k, n)
        return {"logits": np.eye(k)[y] * 6 + rng.normal(size=(n, k)), "labels": y,
                "feats": np.eye(k, 256)[y] * 4 + rng.normal(size=(n, 256)) + shift,
                "pooled": np.eye(k, 768)[y] * 4 + rng.normal(size=(n, 768)) + shift}

    a, b = split(300), split(100, shift=0.5)
    return api.make_ctx(split(3000), {key: np.concatenate([a[key], b[key]]) for key in a})


def cache_ctx(run_dir) -> api.Ctx:
    """Validation context from one run directory on the open volume: cache.npz with train, val and
    oos_val arrays (train labels only). Eval is validation in-scope plus validation OOS, unlabelled."""
    c = np.load(Path(run_dir) / "cache.npz")
    part = lambda s: {k: c[f"{s}__{k}"] for k in ("logits", "feats", "pooled")}
    val, oos = part("val"), part("oos_val")
    return api.make_ctx({**part("train"), "labels": c["train__labels"]},
                        {k: np.concatenate([val[k], oos[k]]) for k in val})


class Proposer:
    def __init__(self, memory, llm, ctx: api.Ctx, out_dir="proposals", model: str = "", checked_on: str = "synthetic",
                 max_attempts: int = 2):
        self.memory, self.llm, self.ctx = memory, llm, ctx
        self.out, self.model, self.checked_on, self.max_attempts = Path(out_dir), model, checked_on, max_attempts
        self.template = (self.out / "_template.py").read_text(encoding="utf-8") if (self.out / "_template.py").exists() else ""
        z = ctx.eval.logits
        self.known = {"energy": ctx.h.energy(z), "msp": ctx.h.msp(z)}      # scores a new proposal must differ from
        self.rows, self.rejected = [], []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        path = self.out / "llm.json"
        if path.exists():                                                   # resume: reload earlier accepted proposals
            old = json.loads(path.read_text(encoding="utf-8"))
            self.rows, self.rejected = old.get("proposals", []), old.get("rejected", [])
            for row in self.rows:
                try:
                    self.known[row["sha"]] = api.run_proposal(api.load_proposal(self.out / f"{row['sha']}.py"), ctx)[0]
                except (api.Malformed, FileNotFoundError):
                    pass

    # ------------------------------------------------------------------ one proposal
    def _prompt(self, idx: int) -> str:
        entries = self.memory.read(["source"], limit=8) + self.memory.read(["lesson"], limit=6) \
            + self.memory.read(["verdict"], limit=12) + self.memory.read(["proposal"], limit=40)
        recalled = ""
        if hasattr(self.memory, "recall"):          # past.dev: ranked evidence across everything any agent wrote
            hits = self.memory.recall(f"What has been tried or learned about: {ANGLES[idx % len(ANGLES)]}")
            if hits:
                recalled = "\n\n## Recalled from shared memory (past.dev)\n" + "\n".join(
                    f"- {' '.join(h['ids']) or '(no id)'}: {h['text']}" for h in hits[:6])
        return "\n\n".join([
            "## API\n" + API_DOC,
            "## Example proposal\n```python\n" + self.template + "```",
            "## Memory (cite the ids you use in MEMORY_USED; SOURCE may only be a url from a source card)\n"
            + format_for_prompt(sorted(entries, key=lambda e: e["id"])) + recalled,
            "## Direction for this proposal\n" + ANGLES[idx % len(ANGLES)],
            "Write the proposal file now.",
        ])

    def _check(self, text: str) -> tuple[str, np.ndarray]:
        """All gates before a file enters the stream. Returns (sha, scores). Raises Malformed."""
        header = api.static_check(text)
        unknown = set(header["MEMORY_USED"]) - self.memory.ids()
        if unknown:
            raise api.Malformed("memory_ids", f"MEMORY_USED lists ids that are not in memory: {sorted(unknown)}")
        urls = {e["content"].get("url") for e in self.memory.read(["source"])}
        if header["SOURCE"] and header["SOURCE"] not in urls:
            raise api.Malformed("source", "SOURCE must be the url of a source card in memory, or empty")
        sha = api.proposal_sha(text)
        if (self.out / f"{sha}.py").exists():
            raise api.Malformed("duplicate", "this exact file already exists")
        tmp = self.out / f"{sha}.py"
        tmp.write_text(text, encoding="utf-8")
        try:
            scores = api.validate(api.load_proposal(tmp), self.ctx)["scores"]
            for name, other in self.known.items():
                rho = spearmanr(scores, other).statistic
                if rho >= DISTINCT:
                    raise api.Malformed("duplicate", f"Spearman {rho:.4f} with {name}: this ranks items the same way")
        except api.Malformed:
            tmp.unlink()
            raise
        return sha, scores

    def propose_one(self, idx: int) -> dict:
        user, feedback, last = self._prompt(idx), "", None
        for attempt in range(1, self.max_attempts + 1):
            reply, usage = self.llm(SYSTEM, user + feedback)
            self.usage["calls"] += 1
            for key in ("input_tokens", "output_tokens"):
                self.usage[key] += usage.get(key, 0)
            text = extract_code(reply)
            try:
                sha, scores = self._check(text)
            except api.Malformed as e:
                last = e
                feedback = (f"\n\n## Your previous file was rejected\n{e.reason}: {e.detail}\n"
                            f"```python\n{text}```\nReturn a corrected, complete file.")
                continue
            header = api.static_check(text)
            self.known[sha] = scores
            row = {"sha": sha, "stream": "llm", "idx": idx, "model": self.model, "attempts": attempt,
                   "angle": idx % len(ANGLES), "family": header["FAMILY"], "arm": header["ARM"],
                   "source": header["SOURCE"], "memory_used": header["MEMORY_USED"], "checked_on": self.checked_on}
            row["memory_id"] = self.memory.write("proposal", "proposer", {
                "sha": sha, "family": header["FAMILY"], "arm": header["ARM"], "rationale": header["RATIONALE"],
                "source": header["SOURCE"], "memory_used": header["MEMORY_USED"]})
            self.rows.append(row)
            return {"status": "ok", **row}
        rej = {"idx": idx, "reason": last.reason, "detail": last.detail[:300]}
        self.rejected.append(rej)
        return {"status": "rejected", **rej}

    # ------------------------------------------------------------------ the loop
    def save(self) -> None:
        write_manifest(self.out, "llm.json", {"stream": "llm", "model": self.model, "proposals": self.rows,
                                              "rejected": self.rejected, "usage": self.usage})

    def run(self, n: int, scout=None, scout_every: int = SCOUT_EVERY, budget_tokens: int = 3_000_000) -> None:
        start = len(self.rows) + len(self.rejected)
        for idx in range(start, start + n):
            if scout is not None and idx % scout_every == 0:
                try:
                    ids = scout.search()
                    print(f"[scout] {len(ids)} new source cards: {ids}")
                except Exception as e:  # a failed search must not stop the stream
                    print(f"[scout] search failed: {type(e).__name__}: {e}")
            try:
                out = self.propose_one(idx)
            except Exception as e:      # API hiccup: record it and carry on
                out = {"status": "rejected", "idx": idx, "reason": "llm_error", "detail": f"{type(e).__name__}: {e}"[:300]}
                self.rejected.append({k: out[k] for k in ("idx", "reason", "detail")})
            tag = out.get("sha") or out["reason"]
            print(f"[{idx:03d}] {out['status']:<8} {tag}  ({out.get('family', '')}, attempts {out.get('attempts', '-')})")
            self.save()
            used = self.usage["input_tokens"] + self.usage["output_tokens"]
            if used > budget_tokens:
                print(f"token budget reached ({used:,}); stopping")
                break
        print(f"accepted {len(self.rows)}, rejected {len(self.rejected)}, tokens {self.usage}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default=os.environ.get("SKEPTIC_PROPOSER_MODEL", DEFAULT_MODEL))
    ap.add_argument("--out", default="proposals")
    ap.add_argument("--memory", default="runs/memory.jsonl")
    ap.add_argument("--cache", default=None, help="a run directory from the open volume, containing cache.npz")
    ap.add_argument("--no-scout", action="store_true")
    ap.add_argument("--budget-tokens", type=int, default=3_000_000)
    a = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    llm = openai_llm(a.model)
    mem = get_memory(a.memory)
    print("memory backend:", type(mem).__name__)
    scout = None
    if not a.no_scout:
        from exa_py import Exa
        from skeptic.streams.scout import Scout
        scout = Scout(mem, llm, Exa(os.environ["EXA_API_KEY"]))
    ctx = cache_ctx(a.cache) if a.cache else smoke_ctx()
    Proposer(mem, llm, ctx, a.out, a.model, "validation" if a.cache else "synthetic").run(
        a.n, scout=scout, budget_tokens=a.budget_tokens)
    print("memory writers:", mem.stats())
    if hasattr(mem, "ingested"):
        print(f"past.dev: {mem.ingested} entries ingested, {mem.recalls} recalls, {mem.errors} errors")
