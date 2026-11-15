import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the score for confidently in-scope-like items, while a train-calibrated smooth gate increasingly adds pooled-space Mahalanobis novelty only in the uncertain energy tail. This avoids broadly penalizing confident known-intent queries for benign feature variation."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0065"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_lo, e_hi = np.quantile(train_energy, [0.70, 0.90])
    e_scale = max(float(e_hi - e_lo), 1e-6)
    gate = 1.0 / (1.0 + np.exp(-np.clip((eval_energy - e_lo) / e_scale, -30.0, 30.0)))

    d_mid, d_hi = np.quantile(train_dist, [0.50, 0.75])
    d_scale = max(float(d_hi - d_mid), 1e-6)
    novelty = (eval_dist - d_mid) / d_scale

    scores = eval_energy + gate * novelty
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
