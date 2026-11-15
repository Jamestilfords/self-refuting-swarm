import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train-calibrated z-scores put classifier energy and pooled-space Mahalanobis distance on a common scale. A smooth maximum emphasizes whichever evidence source is unusually OOS-like while preserving complementary feature and logit signals."
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

    def standardize(train_values, values):
        valid = train_values[np.isfinite(train_values)]
        if valid.size == 0:
            return np.zeros(values.shape, dtype=float)
        center = float(np.mean(valid))
        scale = float(np.std(valid))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        clean = np.nan_to_num(
            values,
            nan=center,
            posinf=center + 20.0 * scale,
            neginf=center - 20.0 * scale,
        )
        return np.clip((clean - center) / scale, -30.0, 30.0)

    z_energy = standardize(train_energy, eval_energy)
    z_dist = standardize(train_dist, eval_dist)

    smooth_max = np.logaddexp(z_energy, z_dist)
    agreement = 0.15 * np.tanh(z_energy) * np.tanh(z_dist)
    return smooth_max + agreement
