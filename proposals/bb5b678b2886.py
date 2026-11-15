import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy measures classifier uncertainty while pooled-space Mahalanobis distance measures representation support. Robust train-fitted centering and scaling place both signals on a comparable scale before combining them."
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0001", "m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_distance = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_distance = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def robust_standardize(train_values, values):
        center = np.median(train_values)
        scale = np.median(np.abs(train_values - center)) * 1.4826
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return (values - center) / scale

    e = robust_standardize(train_energy, eval_energy)
    d = robust_standardize(train_distance, eval_distance)
    return np.nan_to_num(0.5 * e + 0.5 * d, nan=0.0, posinf=1e6, neginf=-1e6)
