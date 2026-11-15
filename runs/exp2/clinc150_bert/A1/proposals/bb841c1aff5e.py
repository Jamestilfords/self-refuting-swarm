import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This gate preserves energy ranking for confident in-scope-like predictions, but progressively relies on pooled-feature Mahalanobis distance when energy indicates uncertainty. The gate and both score scales are calibrated only from training items."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    e_med = np.median(train_energy)
    e_iqr = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)
    e_scale = max(float(e_iqr), 1e-6)

    d_med = np.median(train_dist)
    d_iqr = np.quantile(train_dist, 0.75) - np.quantile(train_dist, 0.25)
    d_scale = max(float(d_iqr), 1e-6)

    e_norm = (eval_energy - e_med) / e_scale
    d_norm = (eval_dist - d_med) / d_scale

    uncertainty_anchor = np.quantile(train_energy, 0.70)
    gate = 1.0 / (1.0 + np.exp(-np.clip(
        (eval_energy - uncertainty_anchor) / e_scale, -40.0, 40.0
    )))

    return (1.0 - gate) * e_norm + gate * d_norm
