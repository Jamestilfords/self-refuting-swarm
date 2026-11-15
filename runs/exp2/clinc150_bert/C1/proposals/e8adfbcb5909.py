import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Low-entropy queries retain the energy ranking, while high-entropy queries receive an increasingly strong pooled-feature Mahalanobis penalty. This targets ambiguous logits whose OOS evidence is more visible in representation-space distance."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_d = g.min_mahalanobis_sq(tr.pooled)
    eval_d = g.min_mahalanobis_sq(ev.pooled)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    train_entropy = ctx.h.entropy(tr.logits)
    eval_entropy = ctx.h.entropy(ev.logits)

    e_center = np.median(train_energy)
    e_scale = np.median(np.abs(train_energy - e_center)) + 1e-6
    d_center = np.median(train_d)
    d_scale = np.median(np.abs(train_d - d_center)) + 1e-6

    h_center = np.median(train_entropy)
    h_scale = np.median(np.abs(train_entropy - h_center)) + 1e-6

    energy_z = (eval_energy - e_center) / e_scale
    distance_z = (eval_d - d_center) / d_scale
    uncertainty_gate = 1.0 / (1.0 + np.exp(-np.clip(
        (eval_entropy - h_center) / h_scale, -30.0, 30.0
    )))

    scores = energy_z + uncertainty_gate * distance_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
