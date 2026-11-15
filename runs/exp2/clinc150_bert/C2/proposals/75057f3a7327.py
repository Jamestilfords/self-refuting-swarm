import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Robust train-calibrated scaling makes logit energy and pooled-feature Mahalanobis distance comparable, so queries that are anomalous in either prediction space or representation space receive high scores."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def robust_scale(train_values, values):
        train_values = np.nan_to_num(
            train_values, nan=0.0, posinf=1e6, neginf=-1e6
        )
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
        center = np.median(train_values)
        lo, hi = np.quantile(train_values, [0.1, 0.9])
        scale = hi - lo
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return np.clip((values - center) / scale, -30.0, 30.0)

    energy_z = robust_scale(train_energy, eval_energy)
    distance_z = robust_scale(train_dist, eval_dist)
    return 0.5 * energy_z + 0.5 * distance_z
