import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions emphasize low-variance constraints that in-scope representations satisfy but OOS queries often violate. A train-calibrated energy term complements this representation-space signal with classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    tail_train = np.asarray(partial.bottom(tr.pooled), dtype=float).reshape(-1)
    tail_eval = np.asarray(partial.bottom(ev.pooled), dtype=float).reshape(-1)

    energy_train = np.asarray(ctx.h.energy(tr.logits), dtype=float).reshape(-1)
    energy_eval = np.asarray(ctx.h.energy(ev.logits), dtype=float).reshape(-1)

    def robust_standardize(train_values, values):
        center = np.median(train_values)
        scale = np.median(np.abs(train_values - center))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return np.clip((values - center) / scale, -1e6, 1e6)

    tail_z = robust_standardize(tail_train, tail_eval)
    energy_z = robust_standardize(energy_train, energy_eval)
    return np.asarray(tail_z + 0.35 * energy_z, dtype=float)
