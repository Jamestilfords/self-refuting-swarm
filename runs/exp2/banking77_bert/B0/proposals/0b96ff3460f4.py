import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS queries are often both far from all intent centroids and weakly separated between "
    "their two nearest intent regions. A train-calibrated combination of novelty and low "
    "nearest-versus-second-nearest distance separation captures this class-boundary ambiguity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0028"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]

    train_near = train_two[:, 0]
    eval_near = eval_two[:, 0]
    train_second = train_two[:, 1]
    eval_second = eval_two[:, 1]

    eps = 1e-8
    train_sep = (train_second - train_near) / (train_second + eps)
    eval_sep = (eval_second - eval_near) / (eval_second + eps)

    near_center = np.median(train_near)
    near_scale = np.median(np.abs(train_near - near_center)) + eps
    sep_center = np.median(train_sep)
    sep_scale = np.median(np.abs(train_sep - sep_center)) + eps

    novelty = (eval_near - near_center) / near_scale
    ambiguity = (sep_center - eval_sep) / sep_scale
    return np.nan_to_num(novelty + 0.5 * ambiguity, nan=0.0, posinf=1e12, neginf=-1e12)
