import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy reflects classifier uncertainty while bottleneck Mahalanobis distance measures deviation from known intent geometry. Standardizing both scores with training-only location and scale makes their complementary OOS signals comparable before fusion."
SOURCE = ""
PARENTS = ["energy", "d_port"]
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    train_d = g.min_mahalanobis_sq(tr.feats)
    eval_d = g.min_mahalanobis_sq(ev.feats)

    train_e = ctx.h.energy(tr.logits)
    eval_e = ctx.h.energy(ev.logits)

    e_center = np.median(train_e)
    d_center = np.median(train_d)
    e_scale = np.median(np.abs(train_e - e_center)) * 1.4826
    d_scale = np.median(np.abs(train_d - d_center)) * 1.4826

    if not np.isfinite(e_scale) or e_scale < 1e-8:
        e_scale = np.std(train_e) + 1e-8
    if not np.isfinite(d_scale) or d_scale < 1e-8:
        d_scale = np.std(train_d) + 1e-8

    energy_z = np.clip((eval_e - e_center) / e_scale, -20.0, 20.0)
    distance_z = np.clip((eval_d - d_center) / d_scale, -20.0, 20.0)
    return 0.5 * energy_z + 0.5 * distance_z
