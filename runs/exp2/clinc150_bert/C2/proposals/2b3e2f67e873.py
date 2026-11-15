import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries tend to be both far from the class selected by the classifier and geometrically ambiguous between nearby class distributions. This score combines predicted-class Mahalanobis mismatch with a small nearest-versus-second-nearest distance gap."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    train_pred = np.argmax(tr.logits, axis=1)
    train_pred_d = train_d[np.arange(len(train_pred)), train_pred]
    train_sorted = np.partition(train_d, 1, axis=1)
    train_gap = train_sorted[:, 1] - train_sorted[:, 0]

    d_center = np.median(train_pred_d)
    d_scale = np.median(np.abs(train_pred_d - d_center)) + 1e-6
    gap_center = np.median(train_gap)
    gap_scale = np.median(np.abs(train_gap - gap_center)) + 1e-6

    eval_d = g.class_mahalanobis_sq(ev.pooled)
    eval_pred = np.argmax(ev.logits, axis=1)
    eval_pred_d = eval_d[np.arange(len(eval_pred)), eval_pred]
    eval_sorted = np.partition(eval_d, 1, axis=1)
    eval_gap = eval_sorted[:, 1] - eval_sorted[:, 0]

    distance_term = (eval_pred_d - d_center) / d_scale
    ambiguity_term = (gap_center - eval_gap) / gap_scale
    return distance_term + ambiguity_term
