import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries often have weak separation between their two closest class manifolds, especially when the classifier's predicted class disagrees with the representation-space nearest class. This score combines low class-distance margin with classifier-manifold disagreement while removing the overall distance level that caused duplication with full Mahalanobis density."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_order = np.partition(train_d, 1, axis=1)
    eval_order = np.partition(eval_d, 1, axis=1)
    train_margin = train_order[:, 1] - train_order[:, 0]
    eval_margin = eval_order[:, 1] - eval_order[:, 0]

    scale = np.median(train_margin)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = np.mean(np.abs(train_margin)) + 1e-6

    pred = np.argmax(ev.logits, axis=1)
    nearest = np.argmin(eval_d, axis=1)
    pred_d = eval_d[np.arange(eval_d.shape[0]), pred]
    nearest_d = eval_d[np.arange(eval_d.shape[0]), nearest]

    disagreement = np.maximum(pred_d - nearest_d, 0.0) / (scale + eval_margin)
    ambiguity = scale / (scale + eval_margin)
    scores = ambiguity + disagreement

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
