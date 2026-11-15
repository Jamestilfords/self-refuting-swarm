import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A train-calibrated gate preserves energy for confident, low-energy queries while progressively emphasizing encoder-space Mahalanobis distance for energy-uncertain queries. This targets cases where logit uncertainty alone is ambiguous but feature support is weak."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    gda = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(gda.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(gda.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_med = np.median(train_energy)
    d_med = np.median(train_dist)
    e_scale = np.percentile(train_energy, 75) - np.percentile(train_energy, 25)
    d_scale = np.percentile(train_dist, 75) - np.percentile(train_dist, 25)
    e_scale = max(float(e_scale), 1e-6)
    d_scale = max(float(d_scale), 1e-6)

    e_train_z = (train_energy - e_med) / e_scale
    gate_center = np.percentile(e_train_z, 60)
    e_z = np.clip((eval_energy - e_med) / e_scale, -30.0, 30.0)
    d_z = np.clip((eval_dist - d_med) / d_scale, -30.0, 30.0)

    uncertainty_gate = 1.0 / (1.0 + np.exp(-3.0 * (e_z - gate_center)))
    scores = (1.0 - uncertainty_gate) * e_z + uncertainty_gate * d_z
    return np.nan_to_num(scores, nan=0.0, posinf=30.0, neginf=-30.0)
