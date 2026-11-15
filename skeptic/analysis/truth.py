"""True improvement, false discoveries, recall, and final-pick quality.

This module reads scores and decision rows. It does not call ``verdict``, the
ratchet, or Thresholdout. The first real pair in a process is checked against
ER-1's paired bootstrap at B = 20.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bert_daedl.eval.metrics import ece
from bert_daedl.posthoc.stats import auroc_scores, combined_interval, fpr_at_95
from skeptic.analysis.bootstrap import paired_bootstrap_fast
from skeptic.analysis.config import DEFAULT, ER1_COMMIT, AnalysisConfig
from skeptic.analysis.decisions import canary_mask, eligible_shas
from skeptic.analysis.sources import ScoreSource

logger = logging.getLogger(__name__)

PairKey = tuple[str, str, str]
_SELF_CHECKED = False


@dataclass(frozen=True)
class TruthResult:
    candidate: str
    incumbent: str
    cell: str
    delta_test: float
    seed_deltas: tuple[float, ...]
    se_boot: float
    se_combined: float
    ci95: tuple[float, float]
    true_primary: bool
    true_strict: bool | None
    strict_detail: dict
    B: int
    seed: int


@dataclass(frozen=True)
class FDPResult:
    n_accepted: int
    n_false: int
    fdp: float | None


@dataclass(frozen=True)
class RecallResult:
    n_true_in_stream: int
    n_true_accepted: int
    recall: float | None


@dataclass(frozen=True)
class WCResult:
    n: int
    mean_gap: float | None


def _ensure_bootstrap_equivalence(id_a, oos_a, id_b, oos_b, seed: int) -> None:
    """One ER-1 comparison per process, on the first real pair."""
    global _SELF_CHECKED
    if _SELF_CHECKED:
        return
    from skeptic.analysis.bootstrap import paired_bootstrap_fast as real_fast

    if real_fast is not paired_bootstrap_fast:
        return
    from bert_daedl.posthoc.stats import paired_bootstrap

    fast = real_fast(id_a, oos_a, id_b, oos_b, B=20, seed=seed, hierarchical=False)
    slow = paired_bootstrap(id_a, oos_a, id_b, oos_b, B=20, seed=seed, hierarchical=False)
    for key in ("delta", "ci_low", "ci_high", "se", "p"):
        if float(fast[key]) != float(slow[key]):
            raise AssertionError(
                f"paired_bootstrap_fast diverged from ER-1 on {key}: {fast[key]} != {slow[key]}"
            )
    _SELF_CHECKED = True


def _check_definition(config: AnalysisConfig) -> None:
    if config.truth_ci_level != 0.95:
        raise ValueError("primary truth is defined at the combined 95% interval")
    if config.strict_reversal != "ci95_high_below_0":
        raise ValueError("strict reversal is ci95_high < 0")
    if config.recall_numerator != "stream_truth_set":
        raise ValueError("recall numerator is membership in the stream truth set")


def seed_deltas(candidate: str, incumbent: str, cell: str, source: ScoreSource) -> np.ndarray:
    id_a, oos_a = source.get(cell, candidate)
    id_b, oos_b = source.get(cell, incumbent)
    if len(id_a) != len(id_b):
        raise ValueError(f"{candidate} and {incumbent} have different seed counts in {cell}")
    return np.asarray([
        auroc_scores(id_a[index], oos_a[index]) - auroc_scores(id_b[index], oos_b[index])
        for index in range(len(id_a))
    ], dtype=np.float64)


def pair_cache_key(
    source: ScoreSource,
    candidate: str,
    incumbent: str,
    cell: str,
    config: AnalysisConfig,
    strict: bool,
) -> str:
    raw = "|".join([
        source.content_hash(cell, candidate),
        source.content_hash(cell, incumbent),
        cell,
        str(config.test_B),
        str(config.test_seed),
        ER1_COMMIT,
    ])
    if strict:
        extra = []
        for other in config.cells:
            if other == cell:
                continue
            cand = (
                source.content_hash(other, candidate)
                if source.has(other, candidate)
                else f"missing {candidate}"
            )
            inc = (
                source.content_hash(other, incumbent)
                if source.has(other, incumbent)
                else f"missing {incumbent}"
            )
            extra.append(f"{other}:{cand}:{inc}")
        raw += "|strict|" + "|".join(extra)
    return hashlib.sha256(raw.encode()).hexdigest()


def _result_to_json(result: TruthResult) -> dict:
    return {
        "candidate": result.candidate,
        "incumbent": result.incumbent,
        "cell": result.cell,
        "delta_test": result.delta_test,
        "seed_deltas": list(result.seed_deltas),
        "se_boot": result.se_boot,
        "se_combined": result.se_combined,
        "ci95": [result.ci95[0], result.ci95[1]],
        "true_primary": result.true_primary,
        "true_strict": result.true_strict,
        "strict_detail": result.strict_detail,
        "B": result.B,
        "seed": result.seed,
    }


def _result_from_json(payload: dict) -> TruthResult:
    return TruthResult(
        candidate=str(payload["candidate"]),
        incumbent=str(payload["incumbent"]),
        cell=str(payload["cell"]),
        delta_test=float(payload["delta_test"]),
        seed_deltas=tuple(float(value) for value in payload["seed_deltas"]),
        se_boot=float(payload["se_boot"]),
        se_combined=float(payload["se_combined"]),
        ci95=(float(payload["ci95"][0]), float(payload["ci95"][1])),
        true_primary=bool(payload["true_primary"]),
        true_strict=None if payload["true_strict"] is None else bool(payload["true_strict"]),
        strict_detail=payload["strict_detail"],
        B=int(payload["B"]),
        seed=int(payload["seed"]),
    )


class TruthCache:
    """One JSON file per pair under ``root/<key[:2]>/<key>.json``."""

    def __init__(self, root: str | Path = "runs/truth"):
        self.root = Path(root)

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> TruthResult | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        return _result_from_json(json.loads(path.read_text(encoding="utf-8")))

    def put(self, key: str, result: TruthResult) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(_result_to_json(result), sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)


def true_improvement(
    candidate: str,
    incumbent: str,
    cell: str,
    strict: bool = False,
    *,
    source: ScoreSource,
    config: AnalysisConfig = DEFAULT,
    cache: TruthCache | None = None,
) -> TruthResult:
    """Combined-interval test of test AUROC(candidate) − AUROC(incumbent)."""
    _check_definition(config)
    if getattr(source, "kind", None) != "test":
        raise ValueError("true_improvement requires a test source; use delta_val for validation")
    key = None
    if cache is not None:
        key = pair_cache_key(source, candidate, incumbent, cell, config, strict)
        cached = cache.get(key)
        if cached is not None:
            return cached
    id_a, oos_a = source.get(cell, candidate)
    id_b, oos_b = source.get(cell, incumbent)
    _ensure_bootstrap_equivalence(id_a, oos_a, id_b, oos_b, config.test_seed)
    bootstrap = paired_bootstrap_fast(
        id_a,
        oos_a,
        id_b,
        oos_b,
        B=config.test_B,
        seed=config.test_seed,
        hierarchical=False,
    )
    per_seed = tuple(float(value) for value in seed_deltas(candidate, incumbent, cell, source))
    interval = combined_interval(bootstrap["delta"], bootstrap["se"], per_seed)
    ci95 = (float(interval["ci95_low"]), float(interval["ci95_high"]))
    primary = ci95[0] > 0
    detail: dict = {}
    true_strict: bool | None = None
    if strict:
        reversal = False
        for other in config.cells:
            if other == cell:
                continue
            if not source.has(other, candidate):
                detail[other] = {"undefined": f"missing {candidate}"}
                continue
            if not source.has(other, incumbent):
                detail[other] = {"undefined": f"missing {incumbent}"}
                continue
            other_result = true_improvement(
                candidate,
                incumbent,
                other,
                False,
                source=source,
                config=config,
                cache=cache,
            )
            reversed_cell = other_result.ci95[1] < 0
            detail[other] = {
                "ci95": [other_result.ci95[0], other_result.ci95[1]],
                "reversed": reversed_cell,
            }
            reversal = reversal or reversed_cell
        true_strict = bool(primary and not reversal)
    result = TruthResult(
        candidate=candidate,
        incumbent=incumbent,
        cell=cell,
        delta_test=float(bootstrap["delta"]),
        seed_deltas=per_seed,
        se_boot=float(bootstrap["se"]),
        se_combined=float(interval["se_combined"]),
        ci95=ci95,
        true_primary=primary,
        true_strict=true_strict,
        strict_detail=detail,
        B=int(config.test_B),
        seed=int(config.test_seed),
    )
    if cache is not None and key is not None:
        cache.put(key, result)
    return result


def truth_by_oos_type(
    candidate: str,
    incumbent: str,
    cell: str,
    *,
    source: ScoreSource,
    config: AnalysisConfig = DEFAULT,
) -> dict:
    """Secondary Banking77 breakdown. Never an input to FDP or recall."""
    if getattr(source, "kind", None) != "test":
        raise ValueError("truth_by_oos_type requires a test source")
    labels = source.labels_oos_type(cell)
    if labels is None:
        return {}
    labels = np.asarray(labels)
    id_a, oos_a = source.get(cell, candidate)
    id_b, oos_b = source.get(cell, incumbent)
    if any(len(row) != len(labels) for row in oos_a):
        raise ValueError(f"OOS-type labels do not match scores in {cell}")
    out = {}
    for code, name in ((1, "id_oos"), (2, "ood_oos")):
        mask = labels == code
        if int(np.sum(mask)) == 0:
            out[name] = {"undefined": "empty subset"}
            continue
        sub_a = [np.ascontiguousarray(row[mask]) for row in oos_a]
        sub_b = [np.ascontiguousarray(row[mask]) for row in oos_b]
        bootstrap = paired_bootstrap_fast(
            id_a, sub_a, id_b, sub_b, B=config.test_B, seed=config.test_seed, hierarchical=False,
        )
        per_seed = [
            auroc_scores(id_a[index], sub_a[index]) - auroc_scores(id_b[index], sub_b[index])
            for index in range(len(id_a))
        ]
        interval = combined_interval(bootstrap["delta"], bootstrap["se"], per_seed)
        out[name] = {
            "delta_test": float(bootstrap["delta"]),
            "ci95": [float(interval["ci95_low"]), float(interval["ci95_high"])],
            "true_primary": float(interval["ci95_low"]) > 0,
        }
    return out


def delta_val(candidate: str, incumbent: str, cell: str, source_val: ScoreSource) -> float:
    """Seed-mean validation AUROC gain against the incumbent at acceptance."""
    if getattr(source_val, "kind", None) != "val":
        raise ValueError("delta_val requires a validation source")
    return float(seed_deltas(candidate, incumbent, cell, source_val).mean())


def _proposal_shas(shas, cell: str, stream: str | None) -> list[str]:
    if isinstance(shas, pd.DataFrame):
        frame = shas
        if "cell" in frame.columns:
            frame = frame.loc[frame["cell"].astype(str) == str(cell)]
        if "status" in frame.columns:
            frame = frame.loc[frame["status"].astype(str) == "ok"]
        if not frame.empty:
            frame = frame.loc[~canary_mask(frame)]
        if stream is not None and "stream" in frame.columns:
            frame = frame.loc[frame["stream"].astype(str) == stream]
        if frame.empty or "sha" not in frame.columns:
            return []
        return list(dict.fromkeys(frame["sha"].astype(str)))
    return [str(sha) for sha in shas]


def stream_truth_set(
    cell: str,
    shas,
    *,
    source: ScoreSource,
    config: AnalysisConfig = DEFAULT,
    cache: TruthCache | None = None,
    stream: str | None = None,
) -> tuple[frozenset[str], dict[str, TruthResult]]:
    """Proposals whose test gain over the recall reference clears zero."""
    _check_definition(config)
    found: list[str] = []
    results: dict[str, TruthResult] = {}
    reference = config.recall_reference
    for sha in _proposal_shas(shas, cell, stream):
        if sha == reference:
            continue
        if not source.has(cell, sha) or not source.has(cell, reference):
            continue
        result = true_improvement(
            sha, reference, cell, False, source=source, config=config, cache=cache,
        )
        results[sha] = result
        if result.true_primary:
            found.append(sha)
    return frozenset(found), results


def _pair_key(row) -> PairKey:
    return (str(row.sha), str(row.incumbent), str(row.cell))


def fdp(decisions: pd.DataFrame, truth: Mapping[PairKey, TruthResult], *, strict: bool = False) -> FDPResult:
    """FDP of one rule, cell, and ordering. Undefined when nothing was accepted."""
    n_accepted = int(len(decisions))
    if n_accepted == 0:
        return FDPResult(0, 0, None)
    n_false = 0
    for row in decisions.itertuples(index=False):
        key = _pair_key(row)
        try:
            result = truth[key]
        except KeyError as exc:
            raise KeyError(f"no truth result for {key}") from exc
        flag = result.true_strict if strict else result.true_primary
        if strict and flag is None:
            raise ValueError(f"strict truth was not computed for {key}")
        if not flag:
            n_false += 1
    return FDPResult(n_accepted, n_false, n_false / n_accepted)


def recall(decisions: pd.DataFrame, truth_set: frozenset[str]) -> RecallResult:
    """Share of S_cell that was accepted. Undefined when S_cell is empty."""
    truth_set = frozenset(truth_set)
    n_true = len(truth_set)
    accepted_shas = set() if len(decisions) == 0 else set(decisions["sha"].astype(str))
    n_hit = len(accepted_shas & set(truth_set))
    value = None if n_true == 0 else n_hit / n_true
    return RecallResult(n_true, n_hit, value)


def winners_curse(decisions: pd.DataFrame) -> WCResult:
    """Mean of Δval − Δtest. Change-weighted when each row is one acceptance."""
    n_changes = int(len(decisions))
    if n_changes == 0:
        return WCResult(0, None)
    gap = decisions["delta_val"].to_numpy(dtype=np.float64) - decisions["delta_test"].to_numpy(dtype=np.float64)
    if "weight" in decisions.columns:
        weights = decisions["weight"].to_numpy(dtype=np.float64)
        if float(weights.sum()) == 0.0:
            return WCResult(n_changes, None)
        return WCResult(n_changes, float(np.average(gap, weights=weights)))
    return WCResult(n_changes, float(gap.mean()))


def mean_defined(values) -> tuple[float | None, int]:
    """Mean of the defined entries, and how many were excluded."""
    defined = []
    excluded = 0
    for value in values:
        if value is None:
            excluded += 1
            continue
        number = float(value)
        if not np.isfinite(number):
            excluded += 1
            continue
        defined.append(number)
    if not defined:
        return None, excluded
    return float(np.mean(defined)), excluded


def attach_truth(
    decisions: pd.DataFrame,
    truth: Mapping[PairKey, TruthResult],
    val_source: ScoreSource,
) -> pd.DataFrame:
    """Add recomputed validation gain and the test truth flags."""
    out = decisions.copy()
    if out.empty:
        out["delta_val"] = []
        out["delta_test"] = []
        out["true_primary"] = []
        out["true_strict"] = []
        return out
    memo: dict[PairKey, float] = {}
    delta_vals = []
    delta_tests = []
    primary = []
    strict_flags = []
    for row in out.itertuples(index=False):
        key = _pair_key(row)
        result = truth[key]
        if key not in memo:
            memo[key] = delta_val(key[0], key[1], key[2], val_source)
        delta_vals.append(memo[key])
        delta_tests.append(result.delta_test)
        primary.append(result.true_primary)
        strict_flags.append(result.true_strict)
    out["delta_val"] = delta_vals
    out["delta_test"] = delta_tests
    out["true_primary"] = primary
    out["true_strict"] = strict_flags
    return out


def plan_pairs(
    accepted_df: pd.DataFrame,
    ledger_df: pd.DataFrame,
    *,
    config: AnalysisConfig = DEFAULT,
    strict: bool = True,
) -> list[PairKey]:
    """Accepted pairs, their other-cell copies, and every proposal against the reference."""
    keys: set[PairKey] = set()
    if accepted_df is not None and len(accepted_df):
        for row in accepted_df.itertuples(index=False):
            pair = (str(row.sha), str(row.incumbent), str(row.cell))
            keys.add(pair)
            if strict:
                for other in config.cells:
                    if other != pair[2]:
                        keys.add((pair[0], pair[1], other))
    for cell in config.cells:
        for sha in eligible_shas(ledger_df, cell):
            if sha == config.recall_reference:
                continue
            keys.add((sha, config.recall_reference, cell))
    return sorted(keys)


def _pair_job(job: tuple):
    candidate, incumbent, cell, strict, source, config, cache = job
    if not source.has(cell, candidate) or not source.has(cell, incumbent):
        return (candidate, incumbent, cell), None
    result = true_improvement(
        candidate,
        incumbent,
        cell,
        strict,
        source=source,
        config=config,
        cache=cache,
    )
    return (candidate, incumbent, cell), result


def compute_truth_table(
    pairs,
    *,
    source: ScoreSource,
    config: AnalysisConfig = DEFAULT,
    cache: TruthCache | None = None,
    map_fn=map,
    strict: bool = False,
    strict_pairs: set[PairKey] | None = None,
) -> dict[PairKey, TruthResult]:
    """Evaluate planned pairs. ``map_fn`` may be ``multiprocessing.Pool.imap``."""
    chosen = set() if strict_pairs is None else set(strict_pairs)
    jobs = []
    for candidate, incumbent, cell in pairs:
        use_strict = strict or (candidate, incumbent, cell) in chosen
        jobs.append((candidate, incumbent, cell, use_strict, source, config, cache))
    results: dict[PairKey, TruthResult] = {}
    total = len(jobs)
    for done, (key, result) in enumerate(map_fn(_pair_job, jobs), start=1):
        if result is not None:
            results[key] = result
        if done % 50 == 0 or done == total:
            logger.info("computed %d / %d truth pairs", done, total)
    return results


def _mean_auroc(source: ScoreSource, cell: str, name: str) -> float:
    ids, oos = source.get(cell, name)
    return float(np.mean([auroc_scores(ident, other) for ident, other in zip(ids, oos)]))


def _mean_fpr(source: ScoreSource, cell: str, name: str) -> float:
    ids, oos = source.get(cell, name)
    return float(np.mean([fpr_at_95(ident, other) for ident, other in zip(ids, oos)]))


def _mean_ece(source: ScoreSource, cell: str, name: str, config: AnalysisConfig) -> float | None:
    if config.final_pick_ece != "proposal_probs_else_plain_model":
        raise NotImplementedError("final-pick ECE uses proposal probabilities or the plain model")
    getter = getattr(source, "probabilities", None)
    if getter is not None:
        rows = getter(cell, name)
        if rows:
            values = [
                ece(probs.max(axis=1), probs.argmax(axis=1) == np.asarray(labels))
                for probs, labels in rows
            ]
            return float(np.mean(values))
    plain = getattr(source, "plain_ece", None)
    arm_of = getattr(source, "arm_of", None)
    if plain is None:
        return None
    arm = arm_of(cell, name) if arm_of is not None else None
    if arm is None:
        return None
    return plain(cell, arm)


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def final_pick(
    decisions_all_rows: pd.DataFrame,
    truth_source: ScoreSource,
    *,
    config: AnalysisConfig = DEFAULT,
    val_source: ScoreSource | None = None,
) -> pd.DataFrame:
    """Last incumbent of each ordering, summarised over orderings."""
    if getattr(truth_source, "kind", None) != "test":
        raise ValueError("final_pick scores the sealed test source")
    if val_source is None:
        val_source = getattr(truth_source, "validation", None)
    reference = config.recall_reference
    rows = []
    if decisions_all_rows.empty:
        return pd.DataFrame(rows)
    grouped = decisions_all_rows.groupby(["rule", "cell"], sort=True)
    for (rule, cell), group in grouped:
        picks = []
        for _, ordering_rows in group.groupby("ordering", sort=True):
            taken = ordering_rows.loc[ordering_rows["accepted"].fillna(False).astype(bool)]
            if taken.empty:
                picks.append(reference)
            else:
                last = taken.sort_values(["idx", "sha"]).iloc[-1]
                picks.append(str(last["sha"]))
        aurocs = [_mean_auroc(truth_source, cell, name) for name in picks]
        fprs = [_mean_fpr(truth_source, cell, name) for name in picks]
        eces = [_mean_ece(truth_source, cell, name, config) for name in picks]
        eces_defined = [value for value in eces if value is not None]
        auroc_stats = _distribution(aurocs)
        fpr_stats = _distribution(fprs)
        best_name = ""
        best_val = None
        best_test = None
        if val_source is not None:
            for name in config.density_names:
                if not val_source.has(cell, name):
                    continue
                score = _mean_auroc(val_source, cell, name)
                if best_val is None or score > best_val:
                    best_val = score
                    best_name = name
            if best_name and truth_source.has(cell, best_name):
                best_test = _mean_auroc(truth_source, cell, best_name)
        ref_auroc = _mean_auroc(truth_source, cell, reference) if truth_source.has(cell, reference) else None
        ece_stats = _distribution(eces_defined) if eces_defined else None
        rows.append({
            "rule": rule,
            "cell": cell,
            "n_orderings": int(group["ordering"].nunique()),
            "auroc_median": auroc_stats["median"],
            "auroc_iqr": auroc_stats["iqr"],
            "auroc_min": auroc_stats["min"],
            "auroc_max": auroc_stats["max"],
            "fpr95_median": fpr_stats["median"],
            "fpr95_iqr": fpr_stats["iqr"],
            "fpr95_min": fpr_stats["min"],
            "fpr95_max": fpr_stats["max"],
            "ece_median": None if ece_stats is None else ece_stats["median"],
            "ece_iqr": None if ece_stats is None else ece_stats["iqr"],
            "ece_min": None if ece_stats is None else ece_stats["min"],
            "ece_max": None if ece_stats is None else ece_stats["max"],
            "ref_ce_energy_auroc": ref_auroc,
            "ref_best_density": best_name or None,
            "ref_best_density_auroc": best_test,
        })
    return pd.DataFrame(rows)


def truth_sets_from_table(
    truth: Mapping[PairKey, TruthResult],
    ledger: pd.DataFrame,
    config: AnalysisConfig = DEFAULT,
) -> dict[str, frozenset[str]]:
    """Rebuild S_cell from pairs already scored against the recall reference."""
    sets = {}
    for cell in config.cells:
        eligible = set(eligible_shas(ledger, cell))
        found = {
            candidate
            for (candidate, incumbent, pair_cell), result in truth.items()
            if pair_cell == cell
            and incumbent == config.recall_reference
            and candidate in eligible
            and result.true_primary
        }
        sets[cell] = frozenset(found)
    return sets
