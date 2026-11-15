import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs often have ambiguous class-conditional geometry or disagree with the classifier's predicted class. This score combines a small nearest-versus-second-nearest distance margin with the excess distance to the logit-predicted class."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    train_gap = train_two[:, 1] - train_two[:, 0]
    gap_scale = np.median(train_gap[train_gap > 0])
    if not np.isfinite(gap_scale) or gap_scale <= 1e-12:
        gap_scale = 1.0

    train_pred = np.argmax(tr.logits, axis=1)
    train_min = np.min(train_d, axis=1)
    train_pred_d = np.take_along_axis(
        train_d, train_pred[:, None], axis=1
    ).ravel()
    excess_scale = np.median(train_pred_d - train_min)
    if not np.isfinite(excess_scale) or excess_scale <= 1e-12:
        excess_scale = gap_scale

    d = g.class_mahalanobis_sq(ev.pooled)
    two = np.partition(d, 1, axis=1)[:, :2]
    nearest = two[:, 0]
    gap = np.maximum(two[:, 1] - nearest, 0.0)

    pred = np.argmax(ev.logits, axis=1)
    pred_d = np.take_along_axis(d, pred[:, None], axis=1).ravel()
    disagreement = np.maximum(pred_d - nearest, 0.0)

    ambiguity = np.exp(-np.minimum(gap / gap_scale, 100.0))
    score = ambiguity + np.log1p(disagreement / excess_scale)
    return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
