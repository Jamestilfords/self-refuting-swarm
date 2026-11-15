import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "MSP captures classifier uncertainty while bottleneck-space class-conditional distance detects representation shift. Train-fitted ECDF scaling makes the two signals comparable, and their geometric combination prioritizes queries anomalous on both axes."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    train_msp = ctx.h.msp(tr.logits)
    train_dist = g.min_mahalanobis_sq(tr.feats)

    msp_to_unit = ctx.h.normalizer("ecdf", train_msp)
    dist_to_unit = ctx.h.normalizer("ecdf", train_dist)

    msp_score = np.clip(msp_to_unit(ctx.h.msp(ev.logits)), 0.0, 1.0)
    dist_score = np.clip(
        dist_to_unit(g.min_mahalanobis_sq(ev.feats)), 0.0, 1.0
    )

    return np.sqrt((0.05 + 0.95 * msp_score) * (0.05 + 0.95 * dist_score))
