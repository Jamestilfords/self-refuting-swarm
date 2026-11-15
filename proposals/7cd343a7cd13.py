import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "For confident low-energy queries, the score remains essentially an energy ranking; as energy becomes uncertain relative to the training distribution, it increasingly incorporates pooled-space Mahalanobis distance to distinguish unsupported representations."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0010"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_center = np.median(train_energy)
    e_scale = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)
    e_scale = max(float(e_scale), 1e-6)

    d_center = np.median(train_dist)
    d_scale = np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)
    d_scale = max(float(d_scale), 1e-6)

    e_z = (eval_energy - e_center) / e_scale
    d_z = (eval_dist - d_center) / d_scale

    gate = 1.0 / (1.0 + np.exp(-np.clip(e_z - 0.5, -40.0, 40.0)))
    scores = e_z + gate * d_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
