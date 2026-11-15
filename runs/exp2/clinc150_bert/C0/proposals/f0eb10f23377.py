import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Robust train-set calibration places logit energy and pooled-feature Mahalanobis distance on comparable scales, so queries anomalous in either confidence or representation receive high scores."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    train_dist = np.log1p(np.maximum(g.min_mahalanobis_sq(tr.pooled), 0.0))
    eval_dist = np.log1p(np.maximum(g.min_mahalanobis_sq(ev.pooled), 0.0))

    def robust_unit(train_values, values):
        med = np.median(train_values)
        q25, q75 = np.quantile(train_values, [0.25, 0.75])
        scale = q75 - q25
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        z = np.clip((values - med) / scale, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    energy_component = robust_unit(train_energy, eval_energy)
    distance_component = robust_unit(train_dist, eval_dist)
    return energy_component + distance_component
