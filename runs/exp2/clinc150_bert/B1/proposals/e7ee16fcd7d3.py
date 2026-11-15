import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A training-calibrated uncertainty gate preserves energy ranking for confident queries, but progressively substitutes pooled-feature Mahalanobis distance when energy indicates an ambiguous prediction. This targets near-OOS inputs whose logits are uncertain while retaining representation-based evidence."
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

    e_med = np.median(train_energy)
    e_q25, e_q75 = np.quantile(train_energy, [0.25, 0.75])
    e_scale = max(e_q75 - e_q25, 1e-6)

    d_med = np.median(train_dist)
    d_q25, d_q75 = np.quantile(train_dist, [0.25, 0.75])
    d_scale = max(d_q75 - d_q25, 1e-6)

    energy_z = (eval_energy - e_med) / e_scale
    dist_z = (eval_dist - d_med) / d_scale

    gate_arg = np.clip((eval_energy - e_q75) / e_scale, -40.0, 40.0)
    uncertain_gate = 1.0 / (1.0 + np.exp(-gate_arg))

    scores = (1.0 - uncertain_gate) * energy_z + uncertain_gate * dist_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
