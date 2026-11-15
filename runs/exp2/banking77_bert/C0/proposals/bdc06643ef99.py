import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-conditional nearest-neighbor distance measures whether a query resembles training examples from the class the classifier assigns to it. OOS examples can be close to some in-scope data globally while remaining far from the predicted intent's local neighborhood."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(ev.logits.shape[0], dtype=float)

    for c in np.unique(pred):
        idx = np.flatnonzero(pred == c)
        train_idx = np.flatnonzero(tr.labels == c)

        if train_idx.size == 0:
            k = min(17, tr.feats.shape[0])
            out[idx] = ctx.h.knn(tr.feats, ev.feats[idx], k=k, l2=True)
        else:
            k = min(17, train_idx.size)
            out[idx] = ctx.h.knn(
                tr.feats[train_idx],
                ev.feats[idx],
                k=k,
                l2=True,
            )

    return out
