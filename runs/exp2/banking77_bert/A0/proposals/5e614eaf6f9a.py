import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy is retained for confidently supported predictions, while a train-calibrated feature-space mismatch is activated only in the uncertain energy tail. This targets queries whose logits are ambiguous and whose pooled representation is also atypical for every known intent."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0103", "m_0119"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_med = np.median(train_energy)
    e_scale = np.percentile(train_energy, 75) - np.percentile(train_energy, 25)
    e_scale = max(float(e_scale), 1e-6)

    d_med = np.median(train_dist)
    d_scale = np.percentile(train_dist, 75) - np.percentile(train_dist, 25)
    d_scale = max(float(d_scale), 1e-6)

    uncertainty_gate = 1.0 / (1.0 + np.exp(-np.clip(
        (eval_energy - e_med) / e_scale, -30.0, 30.0
    )))
    feature_excess = (eval_dist - d_med) / d_scale

    return eval_energy + uncertainty_gate * feature_excess
