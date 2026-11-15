"""L3: deterministic proposal stream from ER-1's grid (slide 14). No LLM.

Grid: feature space x PCA x Mahalanobis variant x normalizer x coupling x lambda x arm = 896 configurations.
Draw 300 without replacement under seed 0, then add the eight scorecard pairs from slide 17.

    python -m skeptic.streams.templated --out proposals

writes one file per proposal plus proposals/templated.json (stream tag, configuration, scorecard pairs).

† Defaults chosen here, for Ryu to confirm in PREREGISTRATION.md:
  - "partial" uses the top 128 eigen-directions when PCA = 128 and the remaining ones when PCA = none.
  - A coupling normalizes minus the distance, fitted on train, as ER-1 normalizes the log-density.
  - PARENTS is ["energy", "d_port" | "d_full"] only when the density is exactly a registered baseline.
  - The incumbent of each scorecard pair (SCORECARD below).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import numpy as np

from skeptic.streams.common import render, write, write_manifest

FEATURES = ("feats", "pooled")                      # 256-d bottleneck, 768-d encoder output
PCAS = (None, 128)
VARIANTS = ("min_class", "relative", "mahalanobis_pp", "partial")
NORMALIZERS = ("minmax", "qsigmoid", "ecdf")
LAMBDAS = (1, 2, 5, 10, 20, 40, 80, 160)
ARMS = ("ce", "edl")
N_DRAW, SEED = 300, 0

_FEATURE_NAME = {"feats": "bottleneck features", "pooled": "encoder output"}
_VARIANT_NAME = {"gda": "GDA log-density", "min_class": "min-class Mahalanobis", "relative": "relative Mahalanobis",
                 "mahalanobis_pp": "Mahalanobis++", "partial": "partial Mahalanobis"}


@dataclass(frozen=True, order=True)
class Config:
    feature: str
    pca: int | None
    variant: str
    arm: str
    coupling: str = "none"          # none | mul | add_log
    normalizer: str | None = None   # None when coupling is none
    lam: int | None = None          # only for add_log
    whiten: bool = True

    @property
    def registered(self) -> str | None:
        """Name of the registered baseline density this configuration uses, if any."""
        if self.variant == "gda" and self.whiten:
            if (self.feature, self.pca) == ("feats", 128):
                return "d_port"
            if (self.feature, self.pca) == ("pooled", None):
                return "d_full"
        return None


def grid() -> list[Config]:
    """All 896 configurations, in a fixed order."""
    out = []
    for f in FEATURES:
        for p in PCAS:
            for v in VARIANTS:
                for a in ARMS:
                    out.append(Config(f, p, v, a))
                    for n in NORMALIZERS:
                        out.append(Config(f, p, v, a, "mul", n))
                        out += [Config(f, p, v, a, "add_log", n, lam) for lam in LAMBDAS]
    return out


def render_config(c: Config) -> str:
    pca = "no PCA" if c.pca is None else f"PCA {c.pca}"
    what = f"{_VARIANT_NAME[c.variant]} on {_FEATURE_NAME[c.feature]} ({pca})"
    if not c.whiten:
        what += ", no whitening"
    lines = ["tr, ev, h = ctx.train, ctx.eval, ctx.h"]
    if c.variant == "mahalanobis_pp":
        lines.append(f"x_tr, x_ev = h.l2(tr.{c.feature}), h.l2(ev.{c.feature})")
    else:
        lines.append(f"x_tr, x_ev = tr.{c.feature}, ev.{c.feature}")
    if c.variant == "partial":
        lines.append("g = h.partial(x_tr, tr.labels, top_k=128)")
        dist = "g.top" if c.pca == 128 else "g.bottom"
    else:
        lines.append(f"g = h.gda(x_tr, tr.labels, pca={c.pca}, whiten={c.whiten})")
        dist = {"gda": "g.neg_log_density", "min_class": "g.min_mahalanobis_sq",
                "mahalanobis_pp": "g.min_mahalanobis_sq", "relative": "g.relative_mahalanobis_sq"}[c.variant]
    lines.append(f"d_ev = {dist}(x_ev)")
    if c.coupling == "none":
        lines.append("return d_ev")
        return render(family="density", arm=c.arm, rationale=f"ER-1 grid: {what}, {c.arm.upper()} arm.",
                      body="\n".join(lines))
    lines.append(f'norm = h.normalizer("{c.normalizer}", -{dist}(x_tr))')
    lines.append("p_hat = norm(-d_ev)")
    if c.coupling == "mul":
        lines.append("return h.couple_mul(ev.logits, p_hat)")
        how = f"{c.normalizer} x mul"
    else:
        lines.append(f"return h.couple_add_log(ev.logits, p_hat, {float(c.lam)})")
        how = f"{c.normalizer} x add_log, lambda {c.lam}"
    parents = ["energy", c.registered] if c.registered else []
    return render(family="fusion", arm=c.arm, parents=parents, body="\n".join(lines),
                  rationale=f"ER-1 grid: {what}, coupled to the logits by {how}, {c.arm.upper()} arm.")


def render_logit(score: str, arm: str) -> str:
    return render(family="logit", arm=arm, rationale=f"ER-1 logit score: {score}, {arm.upper()} arm.",
                  body=f"return ctx.h.{score}(ctx.eval.logits)")


# Slide 17. Each finding is a fixed pair: candidate against incumbent, both as proposal files.
_VAC, _MSP_CE = ("logit", "vacuity", "edl"), ("logit", "msp", "ce")
_D_PORT, _D_FULL = Config("feats", 128, "gda", "edl"), Config("pooled", None, "gda", "edl")
SCORECARD = [
    (1, "EDL vacuity's gain over CE MSP is a scoring-rule effect", _VAC, _MSP_CE, ["SCORE-MATCHED"], "all"),
    (2, "Whitening after PCA changes nothing", _D_PORT, Config("feats", 128, "gda", "edl", whiten=False), ["INERT"], "all"),
    (3, "qsigmoid x mul on D_port hurts CLINC150 and raises ECE", Config("feats", 128, "gda", "edl", "mul", "qsigmoid"),
     _VAC, ["NO EFFECT", "MISCALIBRATES"], "clinc150"),
    (4, "DAEDL's multiplicative coupling tracks the density alone", Config("feats", 128, "gda", "edl", "mul", "minmax"),
     _D_PORT, ["INERT"], "all"),
    (5, "Dropping PCA improves the density", Config("feats", None, "gda", "edl"), _D_PORT, ["ACCEPT"], "all"),
    (6, "The bottleneck density beats vacuity on Banking77-OOS", _D_PORT, _VAC, ["ACCEPT"], "banking77_oos"),
    (7, "Additive coupling on D_full beats both inputs on Banking77-OOS",
     Config("pooled", None, "gda", "edl", "add_log", "ecdf", 5), _VAC, ["ACCEPT"], "banking77_oos"),
    (8, "Relative Mahalanobis on the encoder output hurts", Config("pooled", None, "relative", "edl"), _D_FULL,
     ["NO EFFECT"], "all"),
]


def _text(item) -> str:
    return render_config(item) if isinstance(item, Config) else render_logit(item[1], item[2])


def _describe(item) -> dict:
    return asdict(item) if isinstance(item, Config) else {"logit_score": item[1], "arm": item[2]}


def build(out_dir="proposals", n: int = N_DRAW, seed: int = SEED) -> dict:
    """Write the drawn stream and the scorecard files. Returns the manifest."""
    g = grid()
    picks = np.sort(np.random.default_rng(seed).choice(len(g), size=n, replace=False))
    proposals: dict[str, dict] = {}

    def add(item, role: str) -> str:
        sha = write(out_dir, _text(item))
        row = proposals.setdefault(sha, {"sha": sha, "stream": "templated", "roles": [], "config": _describe(item)})
        if role not in row["roles"]:
            row["roles"].append(role)
        return sha

    for i in picks:
        add(g[int(i)], "draw")
    scorecard = [{"n": k, "finding": finding, "candidate": add(cand, "scorecard"),
                  "incumbent": add(inc, "scorecard_incumbent"), "expected": expected, "cells": cells}
                 for k, finding, cand, inc, expected, cells in SCORECARD]
    manifest = {"stream": "templated", "seed": seed, "n_drawn": n, "grid_size": len(g),
                "proposals": sorted(proposals.values(), key=lambda r: r["sha"]), "scorecard": scorecard}
    write_manifest(out_dir, "templated.json", manifest)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="proposals")
    ap.add_argument("--n", type=int, default=N_DRAW)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    m = build(a.out, a.n, a.seed)
    drawn = sum("draw" in p["roles"] for p in m["proposals"])
    print(f"grid {m['grid_size']}, drawn {drawn}, files {len(m['proposals'])}, scorecard pairs {len(m['scorecard'])}")
