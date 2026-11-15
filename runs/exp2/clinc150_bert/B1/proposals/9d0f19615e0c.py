import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "A train-calibrated gate preserves the energy ranking for confident in-scope-like items, "
    "but progressively substitutes pooled-feature Mahalanobis novelty when energy enters its uncertain tail."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_energy = np.nan_to_num(train_energy, nan=0.0, posinf=1e6, neginf=-1e6)
    eval_energy = np.nan_to_num(eval_energy, nan=0.0, posinf=1e6, neginf=-1e6)
    train_dist = np.nan_to_num(train_dist, nan=0.0, posinf=1e6, neginf=0.0)
    eval_dist = np.nan_to_num(eval_dist, nan=0.0, posinf=1e6, neginf=0.0)

    order_d = np.argsort(train_dist)
    sorted_d = train_dist[order_d]
    sorted_e = np.sort(train_energy)
    novelty_on_energy_scale = np.interp(
        eval_dist,
        sorted_d,
        sorted_e,
        left=sorted_e[0],
        right=sorted_e[-1],
    )

    center = np.quantile(train_energy, 0.70)
    lo, hi = np.quantile(train_energy, [0.25, 0.75])
    width = max(float(hi - lo), 1e-6)
    gate = 1.0 / (1.0 + np.exp(-np.clip((eval_energy - center) / width, -30.0, 30.0)))

    scores = (1.0 - gate) * eval_energy + gate * novelty_on_energy_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
