import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "OOS queries may lie near the broad in-scope representation manifold while lacking support "
    "from neighbors belonging to their classifier-predicted intent. A ratio to global neighborhood "
    "distance discounts sparse but otherwise in-distribution regions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = tr.labels.astype(int)
    pred = np.argmax(ev.logits, axis=1)
    k_target = 25

    global_k = min(k_target, tr.pooled.shape[0])
    d_global = ctx.h.knn(tr.pooled, ev.pooled, k=global_k, l2=True)

    out = np.empty(ev.pooled.shape[0], dtype=float)
    n_classes = tr.logits.shape[1]

    for c in range(n_classes):
        query_mask = pred == c
        if not np.any(query_mask):
            continue

        class_mask = labels == c
        class_train = tr.pooled[class_mask]
        if class_train.shape[0] == 0:
            out[query_mask] = 1.0
            continue

        class_k = min(k_target, class_train.shape[0])
        d_class = ctx.h.knn(
            class_train,
            ev.pooled[query_mask],
            k=class_k,
            l2=True,
        )
        out[query_mask] = np.log1p(d_class) - np.log1p(d_global[query_mask])

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
