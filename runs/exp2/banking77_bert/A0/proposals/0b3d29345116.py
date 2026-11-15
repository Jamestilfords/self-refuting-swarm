import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query should be compatible with the class selected by the classifier, not merely with some nearby "
    "known class. Class-conditional empirical tail calibration makes unusually large predicted-class "
    "Mahalanobis distances comparable across intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0051"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_selected = train_dist[np.arange(train_dist.shape[0]), train_pred]
    eval_selected = eval_dist[np.arange(eval_dist.shape[0]), eval_pred]

    scores = np.empty(eval_selected.shape[0], dtype=float)
    for c in range(train_dist.shape[1]):
        ref = np.sort(train_selected[train_pred == c])
        mask = eval_pred == c
        if ref.size < 2:
            fallback = np.sort(train_selected)
            pos = np.searchsorted(fallback, eval_selected[mask], side="right")
            scores[mask] = pos / (fallback.size + 1.0)
        else:
            pos = np.searchsorted(ref, eval_selected[mask], side="right")
            scores[mask] = pos / (ref.size + 1.0)

    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
