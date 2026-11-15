import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy and feature-space Mahalanobis distance capture complementary uncertainty signals. Robust train-fitted median/MAD scaling places both signals on a common bounded scale before fusion."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_dist = g.min_mahalanobis_sq(tr.pooled)
    eval_dist = g.min_mahalanobis_sq(ev.pooled)

    e_med = np.median(train_energy)
    d_med = np.median(train_dist)
    e_scale = np.median(np.abs(train_energy - e_med)) * 1.4826
    d_scale = np.median(np.abs(train_dist - d_med)) * 1.4826

    e_scale = max(float(e_scale), 1e-6)
    d_scale = max(float(d_scale), 1e-6)

    e_z = np.clip((eval_energy - e_med) / e_scale, -30.0, 30.0)
    d_z = np.clip((eval_dist - d_med) / d_scale, -30.0, 30.0)

    e_common = 1.0 / (1.0 + np.exp(-e_z))
    d_common = 1.0 / (1.0 + np.exp(-d_z))

    return 0.5 * e_common + 0.5 * d_common
