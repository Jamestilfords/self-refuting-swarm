import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "An OOS query can be forced into a predicted intent while being geometrically atypical for that intent. Predicted-class Mahalanobis distance measures classifier-feature disagreement without letting the query select its closest class."
SOURCE = "https://arxiv.org/html/2510.15202v2"
PARENTS = []
MEMORY_USED = ["m_0086"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    references = []
    for c in range(train_d.shape[1]):
        own = train_d[train_pred == c, c]
        if own.size == 0:
            own = train_d[tr.labels == c, c]
        if own.size == 0:
            own = train_d[:, c]
        references.append(np.sort(own))

    scores = np.empty(eval_d.shape[0], dtype=float)
    for c in range(eval_d.shape[1]):
        mask = eval_pred == c
        if np.any(mask):
            ref = references[c]
            ranks = np.searchsorted(ref, eval_d[mask, c], side="right")
            scores[mask] = ranks / (ref.size + 1.0)

    return scores
