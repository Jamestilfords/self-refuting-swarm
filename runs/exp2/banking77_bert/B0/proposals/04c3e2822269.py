import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS queries should be both far from their nearest intent centroid and less decisively assigned "
    "to that centroid than typical in-scope queries. A train-scaled nearest-versus-runner-up distance "
    "gap discounts well-separated in-scope regions while retaining globally unsupported inputs."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0141"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    train_near = train_two[:, 0]
    train_gap = np.maximum(train_two[:, 1] - train_two[:, 0], 0.0)

    near_scale = np.median(train_near) + 1e-12
    gap_scale = np.median(train_gap) + 0.01 * near_scale + 1e-12

    eval_d = g.class_mahalanobis_sq(ev.pooled)
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]
    nearest = np.maximum(eval_two[:, 0], 0.0)
    gap = np.maximum(eval_two[:, 1] - eval_two[:, 0], 0.0)

    return np.log1p(nearest / near_scale) - gap / gap_scale
