import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy captures classifier-level uncertainty while pooled-space Mahalanobis distance captures representation shift. Robust train-only median/MAD scaling places both signals on comparable scales without using evaluation-set calibration."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0002"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_d = g.min_mahalanobis_sq(tr.pooled)
    eval_d = g.min_mahalanobis_sq(ev.pooled)

    train_e = ctx.h.energy(tr.logits)
    eval_e = ctx.h.energy(ev.logits)

    def robust_scale(train_values, values):
        med = np.median(train_values)
        mad = np.median(np.abs(train_values - med))
        scale = max(1.4826 * mad, 1e-6)
        return np.clip((values - med) / scale, -30.0, 30.0)

    energy_score = robust_scale(train_e, eval_e)
    distance_score = robust_scale(train_d, eval_d)
    return energy_score + distance_score
