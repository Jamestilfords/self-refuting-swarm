"""G3 benefit. Statistics only; verdict() applies LORD++ and decides pass or fail."""

from __future__ import annotations

from skeptic.critic import er1
from skeptic.critic.cache import StatsCache, stats_key
from skeptic.critic.config import CriticConfig
from skeptic.critic.types import ScoreSet


def benefit_stats(
    candidate: ScoreSet,
    comparator: ScoreSet,
    config: CriticConfig,
    cell: str,
    cache: StatsCache | None,
) -> dict:
    seed_deltas = candidate.seed_aurocs() - comparator.seed_aurocs()
    delta = float(seed_deltas.mean())
    base = {
        "delta": delta,
        "seed_deltas": [float(value) for value in seed_deltas],
        "se_boot": None,
        "se_combined": None,
        "ci95": None,
        "ci95_boot": None,
        "p_two_sided_boot": None,
    }
    if delta <= 0.0:
        return base

    key = stats_key(candidate, comparator, cell, config.bootstrap_B, config.bootstrap_seed)
    hit = None if cache is None else cache.get(key)
    if hit is None:
        bootstrap = er1.paired_bootstrap(
            list(candidate.val_id),
            list(candidate.val_oos),
            list(comparator.val_id),
            list(comparator.val_oos),
            B=config.bootstrap_B,
            seed=config.bootstrap_seed,
        )
        if abs(float(bootstrap["delta"]) - delta) > 1e-12:
            raise AssertionError("paired_bootstrap delta disagrees with the seed-mean AUROC gap")
        interval = er1.combined_interval(bootstrap["delta"], bootstrap["se"], seed_deltas)
        payload = {
            "delta": float(bootstrap["delta"]),
            "se": float(bootstrap["se"]),
            "ci_low": float(bootstrap["ci_low"]),
            "ci_high": float(bootstrap["ci_high"]),
            "p": float(bootstrap["p"]),
            "ci95_low": float(interval["ci95_low"]),
            "ci95_high": float(interval["ci95_high"]),
            "se_combined": float(interval["se_combined"]),
        }
        if cache is not None:
            cache.put(key, payload)
    else:
        payload = hit
        if abs(float(payload["delta"]) - delta) > 1e-12:
            raise AssertionError("cached bootstrap delta disagrees with the seed-mean AUROC gap")

    base.update({
        "se_boot": float(payload["se"]),
        "se_combined": float(payload["se_combined"]),
        "ci95": [float(payload["ci95_low"]), float(payload["ci95_high"])],
        "ci95_boot": [float(payload["ci_low"]), float(payload["ci_high"])],
        "p_two_sided_boot": float(payload["p"]),
        "p_one_sided": float(er1.one_sided_p(payload["delta"], payload["se_combined"])),
    })
    return base
