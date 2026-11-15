import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the dominant score for confident predictions, while an energy-derived uncertainty gate increasingly incorporates encoder-space class distance for uncertain predictions. This targets OOS examples whose logits are ambiguous but whose pooled features are far from every intent cluster."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    e_mid = np.median(train_energy)
    e_scale = max(np.percentile(train_energy, 75) - np.percentile(train_energy, 25), 1e-6)
    d_mid = np.median(train_dist)
    d_scale = max(np.percentile(train_dist, 75) - np.percentile(train_dist, 25), 1e-6)

    gate_arg = np.clip((eval_energy - e_mid) / e_scale, -30.0, 30.0)
    uncertainty_gate = 1.0 / (1.0 + np.exp(-gate_arg))

    dist_in_energy_units = (eval_dist - d_mid) * (e_scale / d_scale)
    scores = eval_energy + uncertainty_gate * dist_in_energy_units
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
