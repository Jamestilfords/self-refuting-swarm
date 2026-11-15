"""Binormal proposal stream with planted effects.

ID and OOS items share a latent difficulty drawn once per split. A proposal's
OOS scores are shifted by mu. With noise variance sigma^2,

    AUROC = Phi(mu / sqrt(2 (1 + sigma^2))).

Every true proposal uses one mu, and every null uses the CE-energy mu, so the
sign of mu(candidate) - mu(incumbent) is the planted truth of that pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from skeptic.analysis.sources import InMemorySource


@dataclass(frozen=True)
class SyntheticStream:
    val: InMemorySource
    test: InMemorySource
    ledger: pd.DataFrame
    mu: dict[str, float]
    planted_true: frozenset[str]
    cell: str
    shas: list[str]


def make_stream(
    seed: int = 0,
    n_true: int = 20,
    n_null: int = 80,
    delta_auc: float = 0.03,
    sizes: dict | None = None,
    n_seeds: int = 5,
    cell: str = "synth/bert",
    sigma: float = 1.0,
) -> SyntheticStream:
    if sizes is None:
        sizes = {"val": (600, 60), "test": (2000, 600)}
    if not 0.0 < 0.90 + delta_auc < 1.0:
        raise ValueError("delta_auc must keep the true AUROC inside (0, 1)")
    scale = float(np.sqrt(2.0 * (1.0 + sigma ** 2)))
    mu0 = float(scale * norm.ppf(0.90))
    mu_true = float(scale * norm.ppf(0.90 + delta_auc))
    trues = [f"true_{index:02d}" for index in range(n_true)]
    nulls = [f"null_{index:02d}" for index in range(n_null)]
    shas = trues + nulls
    mu = {sha: mu_true for sha in trues}
    mu.update({sha: mu0 for sha in nulls})
    mu["energy_ce"] = mu0
    rng = np.random.default_rng(seed)
    stored = {"val": {}, "test": {}}
    for split in ("val", "test"):
        n_id, n_oos = sizes[split]
        difficulty_id = rng.normal(0.0, 1.0, size=n_id)
        difficulty_oos = rng.normal(0.0, 1.0, size=n_oos)
        for name in ["energy_ce", *shas]:
            id_rows = []
            oos_rows = []
            for _seed_index in range(n_seeds):
                id_noise = rng.normal(0.0, sigma, size=n_id)
                oos_noise = rng.normal(0.0, sigma, size=n_oos)
                id_rows.append(difficulty_id + id_noise)
                oos_rows.append(difficulty_oos + mu[name] + oos_noise)
            stored[split][(cell, name)] = (id_rows, oos_rows)
    ledger = pd.DataFrame([
        {
            "sha": sha,
            "stream": "templated" if index % 2 == 0 else "llm",
            "family": "synth",
            "arm": "ce",
            "cell": cell,
            "status": "ok",
            "parents": [],
            "canary": False,
        }
        for index, sha in enumerate(shas)
    ])
    return SyntheticStream(
        val=InMemorySource("val", stored["val"]),
        test=InMemorySource("test", stored["test"]),
        ledger=ledger,
        mu=mu,
        planted_true=frozenset(trues),
        cell=cell,
        shas=shas,
    )
