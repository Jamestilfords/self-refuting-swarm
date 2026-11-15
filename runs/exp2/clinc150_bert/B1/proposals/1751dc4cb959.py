import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Predicted-class kNN measures whether an example lies near training examples from the class the classifier assigns to it. This is more sensitive than global kNN to samples that are close to the domain but inconsistent with any specific intent cluster."
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = []
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    pred = np.argmax(ev.logits, axis=1)
    k_target = 12
    out = np.empty(len(x_eval), dtype=float)

    for c in np.unique(pred):
        rows = np.flatnonzero(pred == c)
        members = x_train[tr.labels == c]

        if len(members) == 0:
            members = x_train

        k = min(k_target, len(members))
        values = np.empty(len(rows), dtype=float)

        for start in range(0, len(rows), 256):
            batch_rows = rows[start:start + 256]
            q = x_eval[batch_rows]
            dist_sq = np.maximum(
                0.0,
                2.0 - 2.0 * np.dot(q, members.T)
            )
            nearest = np.partition(dist_sq, k - 1, axis=1)[:, :k]
            values[start:start + len(batch_rows)] = np.mean(nearest, axis=1)

        out[rows] = values

    return out
