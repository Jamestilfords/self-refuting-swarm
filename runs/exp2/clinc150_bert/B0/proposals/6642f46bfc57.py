import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries are often both far from every class prototype and comparatively ambiguous between the closest class prototypes. A train-calibrated nearest-versus-second-nearest Mahalanobis margin captures this class-conditional ambiguity while retaining absolute manifold departure."
SOURCE = "https://www.emergentmind.com/papers/2101.03778"
PARENTS = []
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]

    train_min = train_two[:, 0]
    train_gap = train_two[:, 1] - train_two[:, 0]
    eval_min = eval_two[:, 0]
    eval_gap = eval_two[:, 1] - eval_two[:, 0]

    min_scale = max(float(np.median(train_min)), 1e-8)
    gap_scale = max(float(np.median(train_gap)), 1e-8)

    departure = np.log1p(np.maximum(eval_min, 0.0) / min_scale)
    ambiguity = np.log1p(np.maximum(eval_gap, 0.0) / gap_scale)
    return departure - ambiguity
