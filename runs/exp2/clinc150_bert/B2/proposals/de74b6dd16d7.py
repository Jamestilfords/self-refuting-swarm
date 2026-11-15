import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy is retained unchanged for training-calibrated confident predictions, while an "
    "above-normal energy gate activates pooled-feature distance only for uncertain assignments."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0118", "m_0119", "m_0137", "m_0138"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    energy_gate = float(np.quantile(train_energy, 0.75))
    dist_center = float(np.quantile(train_dist, 0.50))
    dist_scale = float(
        np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)
    )
    dist_scale = max(dist_scale, 1e-8)

    # Only let representation atypicality alter items whose logit energy is
    # already in the uncertain tail of the in-scope training distribution.
    gate = (eval_energy > energy_gate).astype(float)
    excess_distance = np.maximum((eval_dist - dist_center) / dist_scale, 0.0)

    return eval_energy + gate * excess_distance
