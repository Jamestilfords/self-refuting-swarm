"""L6 and L7: Experiment 2, the closed loop, and the Analyst. Owner: @Jamestilfords.

One run is:  proposal -> validation scoring (5 seeds, one cell) -> rule -> incumbent update -> memory -> next proposal

Arms (pre-registration 4.7):
    A  ratchet (R0); the Proposer sees its own kept and discarded history
    B  critic (R2);  the Proposer sees shared memory: source cards, proposals, verdicts and the Analyst's lessons
    C  critic (R2);  the Proposer sees nothing from earlier proposals
Each run has its own memory file, its own LORD++ state and its own incumbent, and starts from CE energy.
Code computes every verdict. The LLMs (Proposer, Scout, Analyst) only propose and summarize.

H5 measure: a wasted proposal is a rejection whose (FAMILY, verdict) pair matches an earlier rejection in the
same run. `report` compares the wasted share in the second half of each run with the first half.

Files per run directory:
    proposals/           the proposal files and llm.json (Proposer)
    memory.jsonl         every memory entry written in this run
    decisions.jsonl      one row per proposal slot
    state.json           incumbent, LORD++ state, progress (for resume)
    scores/<sha>.npz     per-item validation scores (not in git)

Run it with `python -m modal_jobs.exp2` (scoring happens on Modal). This module has no Modal dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from skeptic import api
from skeptic.agent_runtime import BudgetExhausted, atomic_json
from skeptic.memory import KINDS

ARMS = {"A": {"rule": "ratchet", "sees": ("proposal", "verdict")},
        "B": {"rule": "critic", "sees": ("source", "proposal", "verdict", "lesson")},
        "C": {"rule": "critic", "sees": ()}}
ANALYST_EVERY = 25
SCOUT_EVERY = 10

ANALYST_SYSTEM = (
    "You are the Analyst in an automated research loop on out-of-scope intent detection. A Proposer writes "
    "score functions; a statistical critic gives each one a verdict. You never decide a verdict. You read the "
    "verdicts so far and write ONE short lesson for the Proposer.\n"
    "Verdicts: ACCEPT (real gain), INERT (ranking unchanged), SCORE-MATCHED (a known baseline in disguise, or "
    "no better than the best baseline), NO EFFECT (no gain), UNDERPOWERED (gain too small for this validation "
    "set to confirm), MISCALIBRATES (hurts calibration), BUG (impossible gain or forbidden access).\n"
    "Write at most 70 words of plain text: which kind of proposal keeps failing and why, and one direction "
    "that has not been tried. Name the proposal hashes you rely on.")


def use_fast_bootstrap() -> None:
    """Point the critic's G3 at the vectorised bootstrap. Same draws and same numbers as ER-1's
    paired_bootstrap (tests/test_bridge.py); the pre-registration lists this as deviation 5."""
    from skeptic import fastboot
    from skeptic.critic import er1
    er1.paired_bootstrap = fastboot.paired_bootstrap


class ArmMemory:
    """What one arm's Proposer may read. Everything is still written to the run's memory file."""

    def __init__(self, memory, sees):
        self.memory, self.sees = memory, tuple(sees)

    def write(self, kind, author, content):
        return self.memory.write(kind, author, content)

    def read(self, kinds=None, limit=None, reader=None):
        kinds = [k for k in (kinds or KINDS) if k in self.sees]
        return self.memory.read(kinds, limit) if kinds else []

    def ids(self):
        return {e["id"] for e in self.read()}

    def get(self, entry_id):
        return self.memory.get(entry_id)

    def stats(self):
        return self.memory.stats()


class Analyst:
    """Reads the decisions of a run and writes a lesson into memory every `every` judged proposals."""

    def __init__(self, memory, llm, every: int = ANALYST_EVERY):
        self.memory, self.llm, self.every = memory, llm, every

    def due(self, n_judged: int) -> bool:
        return n_judged > 0 and n_judged % self.every == 0

    def write(self, judged: list[dict]):
        counts: dict = {}
        for d in judged:
            key = f"{d['family']} -> {d['verdict']}"
            counts[key] = counts.get(key, 0) + 1
        recent = "\n".join(f"- {d['sha']} [{d['family']}] {d['verdict']}"
                           + (f", validation gain over the incumbent {d['delta_inc']:+.4f}" if d.get("delta_inc") is not None else "")
                           for d in judged[-self.every:])
        user = (f"{len(judged)} proposals judged so far.\n\nCounts by family and verdict:\n"
                + "\n".join(f"- {k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
                + f"\n\nThe most recent {min(self.every, len(judged))}:\n{recent}\n\nWrite the lesson.")
        text, _ = self.llm(ANALYST_SYSTEM, user)
        text = " ".join(str(text).split())[:700]
        if not text:
            return None
        cites = sorted({d["sha"] for d in judged if d["sha"] in text})
        try:
            return self.memory.write("lesson", "analyst", {"text": text, "cites": cites, "after": len(judged)})
        except ValueError:          # the memory refuses any entry that mentions test data
            return None


class Loop:
    """One run of one arm in one cell.

    score(text) -> {"scores": (seeds, n) array, "n_in": int, "ece": list or None}; raises api.Malformed.
    cell: the object from skeptic.critic_inputs.load_cell (baselines and the CE-energy incumbent).
    """

    def __init__(self, arm: str, cell, run_dir, *, llm, score, model: str, memory, scout=None, analyst_llm=None,
                 config=None, scout_every: int = SCOUT_EVERY, analyst_every: int = ANALYST_EVERY,
                 max_attempts: int = 2, log=print):
        from skeptic.critic import DEFAULT_CONFIG, LordPlusPlus, MemoryStatsCache
        from skeptic.streams.proposer import Proposer, smoke_ctx

        if arm not in ARMS:
            raise ValueError(f"arm must be one of {sorted(ARMS)}")
        self.arm, self.rule, self.cell, self.dir, self.log = arm, ARMS[arm]["rule"], cell, Path(run_dir), log
        self.config = config or DEFAULT_CONFIG
        self.memory, self.score, self.scout, self.scout_every = memory, score, scout, scout_every
        self.cache = MemoryStatsCache()
        (self.dir / "scores").mkdir(parents=True, exist_ok=True)
        self.n_in = len(cell.incumbent.val_id[0])
        self.n_total = self.n_in + len(cell.incumbent.val_oos[0])
        self.n_seeds = len(cell.incumbent.val_id)
        self._stash: dict = {}

        view = ArmMemory(memory, ARMS[arm]["sees"])
        self.proposer = Proposer(view, llm, smoke_ctx(), self.dir / "proposals", model, cell.cell, max_attempts,
                                 validator=self._validate)
        # a new proposal must rank differently from every registered baseline and every earlier proposal
        known = {k: v for k, v in self.proposer.known.items() if k not in ("energy", "msp")}
        for name, base in cell.baselines.scores.items():
            known[name] = np.concatenate([base.val_id[0], base.val_oos[0]])
        self.proposer.known = known
        self.analyst = Analyst(memory, analyst_llm, analyst_every) if (analyst_llm and arm == "B") else None

        self.rows = [json.loads(line) for line in (self.dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
                     if line.strip()] if (self.dir / "decisions.jsonl").exists() else []
        self.lord, self.incumbent = LordPlusPlus(self.config.fdr_alpha, self.config.fdr_w0), cell.incumbent
        state = self.dir / "state.json"
        if state.exists():
            saved = json.loads(state.read_text(encoding="utf-8"))
            if (saved["arm"], saved["cell"]) != (arm, cell.cell):
                raise ValueError("This run directory belongs to another arm or cell")
            self.lord = LordPlusPlus.from_dict(saved["lord"])
            if saved["incumbent"] != cell.incumbent.name:
                self.incumbent = self._scoreset(saved["incumbent"])

    # ------------------------------------------------------------------ scoring
    def _load(self, sha: str):
        path = self.dir / "scores" / f"{sha}.npz"
        if sha not in self._stash and path.exists():
            with np.load(path, allow_pickle=False) as z:
                self._stash[sha] = {"scores": np.asarray(z["scores"], np.float64),
                                    "ece": z["ece"].tolist() if "ece" in z.files else None}
        return self._stash.get(sha)

    def _validate(self, text: str) -> np.ndarray:
        """The Proposer's validator: score on real validation data. Returns seed-0 scores for the duplicate check."""
        sha = api.proposal_sha(text)
        got = self._load(sha)
        if got is None:
            out = self.score(text)
            scores = np.asarray(out["scores"], np.float64)
            if scores.shape != (self.n_seeds, self.n_total) or int(out["n_in"]) != self.n_in or not np.isfinite(scores).all():
                raise api.Malformed("length", f"scores have shape {scores.shape}")
            got = {"scores": scores, "ece": out.get("ece")}
            extra = {"ece": np.asarray(got["ece"], np.float64)} if got["ece"] is not None else {}
            np.savez_compressed(self.dir / "scores" / f"{sha}.npz", scores=scores.astype(np.float32), n_in=self.n_in, **extra)
            got["scores"] = scores.astype(np.float32).astype(np.float64)        # exactly what a resume reloads
            self._stash[sha] = got
        return got["scores"][0]

    def _scoreset(self, sha: str):
        from skeptic.critic import ScoreSet
        got = self._load(sha)
        header = api.static_check((self.dir / "proposals" / f"{sha}.py").read_text(encoding="utf-8"))
        parents = tuple(header["PARENTS"])
        if len(parents) != 2 or any(f"{p}_{header['ARM']}" not in self.cell.baselines.scores for p in parents):
            parents = ()            # G6 is defined only for a fusion of two registered scores
        return ScoreSet(name=sha, arm=header["ARM"], val_id=tuple(s[:self.n_in] for s in got["scores"]),
                        val_oos=tuple(s[self.n_in:] for s in got["scores"]), parents=parents,
                        ece=got["ece"], family=header["FAMILY"])

    # ------------------------------------------------------------------ one slot
    def _judge(self, cand) -> dict:
        delta = cand.mean_auroc() - self.incumbent.mean_auroc()
        if self.rule == "ratchet":
            return {"verdict": "KEPT" if delta > 0 else "DISCARDED", "accepted": bool(delta > 0),
                    "summary": f"validation AUROC {delta:+.4f} against the incumbent"}
        from skeptic.critic import verdict
        d = verdict(cand, self.incumbent, self.cell.baselines, self.lord, self.cell.cell, config=self.config,
                    cache=self.cache).to_row()
        failed = next((g for g, outcome in d["gates"].items() if outcome == "fail"), None)
        summary = f"validation AUROC {delta:+.4f} against the incumbent"
        if d["delta_val"] is not None:
            summary += f"; {d['delta_val']:+.4f} against the comparator {d['comparator']}"
        if d["ci"] is not None:
            summary += f", 95% interval [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"
        if failed:
            summary += f"; stopped at gate {failed}"
        return {"verdict": d["verdict"], "accepted": d["verdict"] == "ACCEPT", "summary": summary,
                "critic": {k: d[k] for k in ("comparator", "delta_val", "ci", "p", "alpha_t", "gates", "mdd")}}

    def step(self, idx: int) -> dict:
        if self.scout is not None and idx % self.scout_every == 0:
            try:
                self.scout.search()
            except BudgetExhausted:
                raise
            except Exception as e:      # a failed search must not stop the loop
                self.log(f"scout search failed: {type(e).__name__}")
        out = self.proposer.propose_one(idx)
        row = {"idx": idx, "arm": self.arm, "cell": self.cell.cell, "rule": self.rule, "incumbent": self.incumbent.name}
        if out["status"] != "ok":
            row.update(status="not_scored", reason=out["reason"], verdict=None, accepted=False)
        else:
            cand = self._scoreset(out["sha"])
            judged = self._judge(cand)
            row.update(status="judged", sha=out["sha"], family=out["family"], score_arm=out["arm"],
                       memory_used=out["memory_used"], source=out["source"],
                       val_auroc=round(cand.mean_auroc(), 6),
                       delta_inc=round(cand.mean_auroc() - self.incumbent.mean_auroc(), 6), **judged)
            try:
                row["memory_id"] = self.memory.write("verdict", "critic", {
                    "sha": out["sha"], "family": out["family"], "verdict": judged["verdict"],
                    "summary": judged["summary"], "rule": self.rule})
            except ValueError:
                row["memory_id"] = None
            if judged["accepted"]:
                self.incumbent = cand
        self.rows.append(row)
        with (self.dir / "decisions.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        self.proposer.save()
        atomic_json(self.dir / "state.json", {"arm": self.arm, "cell": self.cell.cell, "incumbent": self.incumbent.name,
                                              "lord": self.lord.to_dict(), "slots": len(self.rows)})
        judged_rows = [r for r in self.rows if r["status"] == "judged"]
        if self.analyst and row["status"] == "judged" and self.analyst.due(len(judged_rows)):
            try:
                lesson = self.analyst.write(judged_rows)
                self.log(f"analyst lesson {lesson}")
            except BudgetExhausted:
                raise
            except Exception as e:
                self.log(f"analyst failed: {type(e).__name__}")
        return row

    def run(self, n: int) -> list[dict]:
        """Continue until n proposal slots exist. Safe to call again after an interruption."""
        for idx in range(len(self.rows), n):
            row = self.step(idx)
            tail = f"{row['verdict']:<14} {row.get('family', ''):<8} {row.get('delta_inc', 0):+.4f}" if row["status"] == "judged" \
                else f"not scored ({row['reason']})"
            self.log(f"[{idx:03d}] {tail}")
        return self.rows


# ---------------------------------------------------------------------- H5 summary
def wasted_flags(rows: list[dict]) -> list[bool]:
    """For each judged row: is it a rejection whose (family, verdict) already occurred among earlier rejections?"""
    seen, flags = set(), []
    for r in rows:
        key = (r["family"], r["verdict"])
        flags.append(not r["accepted"] and key in seen)
        if not r["accepted"]:
            seen.add(key)
    return flags


def summarize_run(run_dir) -> dict | None:
    path = Path(run_dir) / "decisions.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None
    slots = len(rows)
    judged = [r for r in rows if r["status"] == "judged"]
    flags = wasted_flags(judged)
    half = slots / 2

    def share(keep):
        picked = [f for r, f in zip(judged, flags) if keep(r["idx"])]
        return round(sum(picked) / len(picked), 4) if picked else None

    first, second = share(lambda i: i < half), share(lambda i: i >= half)
    verdicts: dict = {}
    for r in judged:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    accepted = [r for r in judged if r["accepted"]]
    return {"run": str(run_dir), "arm": rows[0]["arm"], "cell": rows[0]["cell"], "slots": slots, "judged": len(judged),
            "not_scored": slots - len(judged), "accepted": len(accepted), "wasted": int(sum(flags)),
            "wasted_first_half": first, "wasted_second_half": second,
            "wasted_change": None if first is None or second is None else round(second - first, 4),
            "final_val_auroc": accepted[-1]["val_auroc"] if accepted else None, "verdicts": verdicts}


def report(root="runs/exp2") -> list[dict]:
    """Every run under root, plus the per-arm mean of the H5 change. Descriptive; no interval is claimed."""
    runs = [s for s in (summarize_run(p.parent) for p in sorted(Path(root).glob("**/decisions.jsonl"))) if s]
    print(f"{'run':<34}{'slots':>6}{'judged':>7}{'accept':>7}{'wasted 1st':>11}{'wasted 2nd':>11}{'change':>8}")
    for s in runs:
        fmt = lambda v: "   -" if v is None else f"{v:.2f}"
        print(f"{Path(s['run']).relative_to(root).as_posix():<34}{s['slots']:>6}{s['judged']:>7}{s['accepted']:>7}"
              f"{fmt(s['wasted_first_half']):>11}{fmt(s['wasted_second_half']):>11}{fmt(s['wasted_change']):>8}")
    for cell in sorted({s["cell"] for s in runs}):
        for arm in sorted(ARMS):
            vals = [s["wasted_change"] for s in runs if s["cell"] == cell and s["arm"] == arm and s["wasted_change"] is not None]
            if vals:
                print(f"{cell} arm {arm}: mean change in wasted share {np.mean(vals):+.3f} over {len(vals)} runs")
    if runs:
        atomic_json(Path(root) / "summary.json", runs)
    return runs


if __name__ == "__main__":
    import sys
    report(sys.argv[2] if len(sys.argv) > 2 else "runs/exp2")
