import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A train-calibrated energy gate preserves the energy ranking for confident inputs while progressively adding pooled-feature Mahalanobis distance for inputs whose energy indicates uncertainty. This targets OOS examples that are logit-ambiguous but geometrically far from all intent classes."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_mid = np.median(train_energy)
    e_scale = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)
    e_scale = max(float(e_scale), 1e-6)

    d_mid = np.median(train_dist)
    d_scale = np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)
    d_scale = max(float(d_scale), 1e-6)

    gate_arg = np.clip((eval_energy - e_mid) / e_scale, -30.0, 30.0)
    uncertainty_gate = 1.0 / (1.0 + np.exp(-gate_arg))

    distance_excess = (eval_dist - d_mid) / d_scale
    scores = eval_energy + uncertainty_gate * distance_excess
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
