import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train-calibrated empirical percentiles put predictive energy and class-conditional feature distance on a common scale. A probabilistic-union fusion raises the OOS score when either confidence or representation geometry is atypical."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_energy = np.nan_to_num(
        np.asarray(ctx.h.energy(tr.logits), dtype=float),
        nan=0.0,
        posinf=1e12,
        neginf=-1e12,
    )
    eval_energy = np.nan_to_num(
        np.asarray(ctx.h.energy(ev.logits), dtype=float),
        nan=0.0,
        posinf=1e12,
        neginf=-1e12,
    )
    train_dist = np.nan_to_num(
        np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float),
        nan=0.0,
        posinf=1e12,
        neginf=-1e12,
    )
    eval_dist = np.nan_to_num(
        np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float),
        nan=0.0,
        posinf=1e12,
        neginf=-1e12,
    )

    def percentile(train_values, values):
        ordered = np.sort(train_values)
        return np.searchsorted(ordered, values, side="right").astype(float) / float(
            ordered.size
        )

    energy_p = percentile(train_energy, eval_energy)
    dist_p = percentile(train_dist, eval_dist)

    return 1.0 - (1.0 - energy_p) * (1.0 - dist_p)
