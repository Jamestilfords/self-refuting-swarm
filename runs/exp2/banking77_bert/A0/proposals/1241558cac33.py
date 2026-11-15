import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Classifier energy is retained unchanged for confident items, while high-energy (uncertain) items receive an additional train-calibrated encoder-space atypicality penalty. This conditional use of Mahalanobis distance targets cases where logits alone are ambiguous."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    energy_gate = np.quantile(train_energy, 0.70)
    dist_center = np.median(train_dist)
    dist_scale = np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)
    energy_scale = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)

    dist_scale = max(float(dist_scale), 1e-8)
    energy_scale = max(float(energy_scale), 1e-8)

    atypicality = np.maximum((eval_dist - dist_center) / dist_scale, 0.0)
    uncertain = (eval_energy > energy_gate).astype(float)
    scores = eval_energy + uncertain * energy_scale * atypicality

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
