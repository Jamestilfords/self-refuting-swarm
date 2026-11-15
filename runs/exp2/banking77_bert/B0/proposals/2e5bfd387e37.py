import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident predictions retain the energy ranking, while uncertain predictions receive an "
    "additional train-calibrated Mahalanobis novelty penalty. This targets queries whose logit "
    "uncertainty alone is ambiguous but whose representation is far from known intent regions."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0003", "m_0012", "m_0018"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_q25, e_q75 = np.quantile(train_energy, [0.25, 0.75])
    e_scale = max(float(e_q75 - e_q25), 1e-6)
    uncertainty_gate = 1.0 / (
        1.0 + np.exp(-np.clip((eval_energy - e_q75) / e_scale, -40.0, 40.0))
    )

    d_q25, d_q75 = np.quantile(train_dist, [0.25, 0.75])
    d_scale = max(float(d_q75 - d_q25), 1e-6)
    distance_novelty = (eval_dist - d_q25) / d_scale

    scores = eval_energy + uncertainty_gate * distance_novelty
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
