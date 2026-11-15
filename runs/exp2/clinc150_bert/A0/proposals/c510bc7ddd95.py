import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A train-calibrated soft gate preserves energy-based ordering for confident items, but progressively substitutes pooled-feature novelty when energy indicates uncertainty. This targets uncertain queries whose representation is either unsupported by, or unusually close to, the in-scope feature distribution."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0019", "m_0015"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    gda = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = np.asarray(gda.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(gda.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_energy = np.nan_to_num(train_energy, nan=0.0, posinf=1e6, neginf=-1e6)
    eval_energy = np.nan_to_num(eval_energy, nan=0.0, posinf=1e6, neginf=-1e6)
    train_dist = np.nan_to_num(train_dist, nan=0.0, posinf=1e6, neginf=-1e6)
    eval_dist = np.nan_to_num(eval_dist, nan=0.0, posinf=1e6, neginf=-1e6)

    e_sorted = np.sort(train_energy)
    d_sorted = np.sort(train_dist)
    n = float(len(train_energy))

    e_rank = np.searchsorted(e_sorted, eval_energy, side="right") / n
    d_rank = np.searchsorted(d_sorted, eval_dist, side="right") / n

    q25, q75 = np.quantile(train_energy, [0.25, 0.75])
    scale = max(float(q75 - q25), 1e-6)
    threshold = float(np.quantile(train_energy, 0.70))
    z = np.clip((eval_energy - threshold) / scale, -30.0, 30.0)
    gate = 1.0 / (1.0 + np.exp(-4.0 * z))

    return (1.0 - gate) * e_rank + gate * d_rank
