import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Robust train-fitted standardization puts classifier energy and pooled-space class distance on comparable scales before averaging their independent OOS evidence. This preserves both logit uncertainty and representation-support failures without using evaluation-set calibration."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0054"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def robust_z(train_values, values):
        med = np.median(train_values)
        scale = 1.4826 * np.median(np.abs(train_values - med))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return np.clip((values - med) / scale, -20.0, 20.0)

    z_energy = robust_z(train_energy, eval_energy)
    z_dist = robust_z(train_dist, eval_dist)
    return 0.5 * z_energy + 0.5 * z_dist
