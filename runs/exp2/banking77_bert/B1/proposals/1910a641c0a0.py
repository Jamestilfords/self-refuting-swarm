import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Train-standardizing energy and pooled-space Mahalanobis distance puts classifier uncertainty and representation atypicality on a common scale before combining them. OOS queries should score highly when either or both signals are unusually large relative to in-scope training examples."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0049", "m_0057"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_center = np.median(train_energy)
    d_center = np.median(train_dist)
    e_scale = np.median(np.abs(train_energy - e_center)) * 1.4826
    d_scale = np.median(np.abs(train_dist - d_center)) * 1.4826

    if not np.isfinite(e_scale) or e_scale < 1e-8:
        e_scale = np.std(train_energy) + 1e-8
    if not np.isfinite(d_scale) or d_scale < 1e-8:
        d_scale = np.std(train_dist) + 1e-8

    energy_z = (eval_energy - e_center) / e_scale
    dist_z = (eval_dist - d_center) / d_scale

    # Soft maximum emphasizes cases anomalous in either calibrated modality
    # while retaining evidence when both are moderately unusual.
    score_out = np.logaddexp(energy_z, dist_z)
    return np.nan_to_num(score_out, nan=0.0, posinf=1e6, neginf=-1e6)
