import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-conditional local support measures whether an item lies near examples of the specific intent predicted by the classifier, catching confidently misrouted OOS queries that may still be close to the overall training manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    predicted = np.argmax(ev.logits, axis=1)

    out = np.empty(x_eval.shape[0], dtype=float)
    block = 256

    for cls in np.unique(predicted):
        eval_idx = np.flatnonzero(predicted == cls)
        train_idx = np.flatnonzero(tr.labels == cls)

        if train_idx.size == 0:
            out[eval_idx] = 2.0
            continue

        k = min(8, train_idx.size)
        class_train = x_train[train_idx]

        for start in range(0, eval_idx.size, block):
            idx = eval_idx[start:start + block]
            distances = 1.0 - x_eval[idx] @ class_train.T
            nearest = np.partition(distances, k - 1, axis=1)[:, :k]
            out[idx] = np.mean(nearest, axis=1)

    return np.maximum(out, 0.0)
