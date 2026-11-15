import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This preserves the energy score for confident in-distribution-like predictions, while progressively injecting pooled-feature Mahalanobis distance only when energy indicates uncertainty. The gate is calibrated solely from training energies, so uncertain examples that are also feature-space distant receive a stronger OOS score."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    gda = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_d = gda.min_mahalanobis_sq(tr.pooled)
    eval_d = gda.min_mahalanobis_sq(ev.pooled)

    train_e = ctx.h.energy(tr.logits)
    eval_e = ctx.h.energy(ev.logits)

    e_lo, e_hi = np.quantile(train_e, [0.70, 0.95])
    d_med = np.median(train_d)
    d_scale = np.quantile(train_d, 0.90) - np.quantile(train_d, 0.10)

    e_width = max(float(e_hi - e_lo), 1e-8)
    d_scale = max(float(d_scale), 1e-8)

    uncertainty_gate = np.clip((eval_e - e_lo) / e_width, 0.0, 1.0)
    energy_scale = max(float(np.std(train_e)), 1e-8)
    distance_term = energy_scale * (eval_d - d_med) / d_scale

    return np.asarray(eval_e + uncertainty_gate * distance_term, dtype=float)
