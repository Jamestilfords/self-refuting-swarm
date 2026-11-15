import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Unusual normalized winner dominance captures atypical classifier competition, while "
    "pooled-space neighbor distance detects representations unsupported by known examples. "
    "Train-fitted empirical tail maps place both signals on a common OOS scale."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0275", "m_0277"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def dominance(logits):
        x = np.asarray(logits, dtype=float)
        centered = x - np.mean(x, axis=1, keepdims=True)
        scale = np.sqrt(np.mean(centered * centered, axis=1)) + 1e-8
        return np.max(x, axis=1) / scale

    train_dom = dominance(tr.logits)
    eval_dom = dominance(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_med = np.median(train_dom)
    global_dev = np.abs(train_dom - global_med)
    global_sorted = np.sort(global_dev)

    logit_tail = np.empty(len(eval_dom), dtype=float)
    for cls in np.unique(eval_pred):
        train_mask = train_pred == cls
        eval_mask = eval_pred == cls
        if np.sum(train_mask) >= 8:
            center = np.median(train_dom[train_mask])
            reference = np.sort(np.abs(train_dom[train_mask] - center))
        else:
            center = global_med
            reference = global_sorted
        deviations = np.abs(eval_dom[eval_mask] - center)
        logit_tail[eval_mask] = (
            np.searchsorted(reference, deviations, side="right") / float(len(reference))
        )

    n_train = len(tr.pooled)
    neighbor_rank = min(10, max(1, n_train - 1))
    train_rank = min(n_train, neighbor_rank + 1)

    train_knn = ctx.h.knn(tr.pooled, tr.pooled, k=train_rank, l2=True)
    eval_knn = ctx.h.knn(tr.pooled, ev.pooled, k=neighbor_rank, l2=True)

    train_knn = np.asarray(train_knn, dtype=float)
    eval_knn = np.asarray(eval_knn, dtype=float)
    knn_reference = np.sort(train_knn)
    feature_tail = (
        np.searchsorted(knn_reference, eval_knn, side="right") / float(len(knn_reference))
    )

    return 0.5 * logit_tail + 0.5 * feature_tail
