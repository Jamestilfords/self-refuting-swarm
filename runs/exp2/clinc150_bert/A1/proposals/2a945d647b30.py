import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions explain variation used to discriminate known intents, so residual directions retain "
    "within-intent semantic detail. Queries routed to an intent but lacking class-conditional residual neighbors "
    "are likely out of scope rather than ordinary examples of that intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0109"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    mean = np.mean(x, axis=0)
    classes = np.unique(y)
    class_means = np.vstack([np.mean(x[y == c], axis=0) for c in classes])
    directions = class_means - mean

    _, singular_values, vt = np.linalg.svd(directions, full_matrices=False)
    if singular_values.size:
        tol = max(float(singular_values[0]) * 1e-10, 1e-12)
        rank = int(np.sum(singular_values > tol))
    else:
        rank = 0

    xc = x - mean
    zc = z - mean
    if rank > 0:
        basis = vt[:rank].T
        xr = xc - (xc @ basis) @ basis.T
        zr = zc - (zc @ basis) @ basis.T
    else:
        xr = xc
        zr = zc

    predicted = np.argmax(np.asarray(ev.logits), axis=1)
    scores = np.empty(zr.shape[0], dtype=float)

    global_k = min(20, max(1, xr.shape[0]))
    global_score = ctx.h.knn(xr, zr, k=global_k, l2=True)

    for c in np.unique(predicted):
        query_mask = predicted == c
        train_mask = y == c
        count = int(np.sum(train_mask))
        if count >= 3:
            k = min(20, count)
            scores[query_mask] = ctx.h.knn(
                xr[train_mask],
                zr[query_mask],
                k=k,
                l2=True,
            )
        else:
            scores[query_mask] = global_score[query_mask]

    scores = np.asarray(scores, dtype=float)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
