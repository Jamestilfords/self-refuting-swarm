import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy captures classifier uncertainty while pooled-space Mahalanobis distance measures feature support; robust train-fitted sigmoid calibration places both signals on a common bounded scale before fusion."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0133"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def calibrated(values, center, scale):
        z = np.clip((values - center) / scale, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    e_center = float(np.median(train_energy))
    d_center = float(np.median(train_dist))
    e_scale = float(np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25))
    d_scale = float(np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25))
    e_scale = max(e_scale / 1.349, 1e-6)
    d_scale = max(d_scale / 1.349, 1e-6)

    e = calibrated(np.nan_to_num(eval_energy, nan=e_center, posinf=e_center + 30.0 * e_scale,
                                 neginf=e_center - 30.0 * e_scale), e_center, e_scale)
    d = calibrated(np.nan_to_num(eval_dist, nan=d_center, posinf=d_center + 30.0 * d_scale,
                                 neginf=d_center - 30.0 * d_scale), d_center, d_scale)
    return e + d
