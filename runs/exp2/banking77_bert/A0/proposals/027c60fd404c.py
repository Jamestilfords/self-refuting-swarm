import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope queries should be substantially closer to one class manifold than to competing class manifolds. OOS queries often have a small nearest-versus-second-nearest Mahalanobis gap, indicating ambiguous class-conditional support."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]

    train_gap = train_two[:, 1] - train_two[:, 0]
    eval_gap = eval_two[:, 1] - eval_two[:, 0]

    scale = np.median(np.abs(train_gap - np.median(train_gap)))
    scale = max(float(scale), 1e-6)
    scores = -eval_gap / scale

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
