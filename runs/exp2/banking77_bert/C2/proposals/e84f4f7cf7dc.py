import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy captures classifier uncertainty while Mahalanobis distance captures feature-space atypicality. Robust train-derived scaling puts both signals on comparable units without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    def robust_scale(train_values, values):
        med = np.median(train_values)
        q25, q75 = np.percentile(train_values, [25.0, 75.0])
        scale = max(float(q75 - q25), 1e-6)
        return np.clip((values - med) / scale, -50.0, 50.0)

    energy_z = robust_scale(train_energy, eval_energy)
    dist_z = robust_scale(train_dist, eval_dist)

    return np.asarray(energy_z + dist_z, dtype=float)
