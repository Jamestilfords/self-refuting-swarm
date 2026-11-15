import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS representations are often both far from every class center and ambiguously close to multiple "
    "known intents. A train-calibrated nearest-versus-second-nearest Mahalanobis ambiguity ratio captures both effects."
)
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = []
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_d = g.class_mahalanobis_sq(tr.pooled)
    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    train_gap = np.maximum(train_two[:, 1] - train_two[:, 0], 0.0)
    gap_scale = float(np.median(train_gap))
    gap_scale = max(gap_scale, 1e-8)

    d = g.class_mahalanobis_sq(ev.pooled)
    two = np.partition(d, 1, axis=1)[:, :2]
    nearest = np.maximum(two[:, 0], 0.0)
    gap = np.maximum(two[:, 1] - two[:, 0], 0.0)

    return nearest / (gap + gap_scale)
