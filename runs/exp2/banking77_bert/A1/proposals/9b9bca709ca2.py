import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A query should be locally supported by training examples from the intent predicted by the classifier, not merely by examples from any intent. Class-conditional cosine kNN distance exposes confidently predicted queries that lie outside the local support of their assigned in-scope intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0077"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pred = np.argmax(ev.logits, axis=1)
    scores = np.empty(len(ev.logits), dtype=float)

    for cls in np.unique(pred):
        test_idx = np.flatnonzero(pred == cls)
        train_idx = np.flatnonzero(tr.labels == cls)

        if len(train_idx) >= 3:
            k = min(15, len(train_idx))
            scores[test_idx] = ctx.h.knn(
                tr.pooled[train_idx],
                ev.pooled[test_idx],
                k=k,
                l2=True,
            )
        else:
            scores[test_idx] = ctx.h.knn(
                tr.pooled,
                ev.pooled[test_idx],
                k=min(15, len(tr.pooled)),
                l2=True,
            )

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
