import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A query should have close support among training examples of its classifier-predicted intent, not merely somewhere in the global representation space. Class-conditional pooled-space kNN radii normalized by each intent's leave-one-out support scale highlight samples that are atypical for their assigned intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0132"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    pred = np.argmax(ev.logits, axis=1)
    classes = np.unique(tr.labels)

    scores = np.empty(len(eval_x), dtype=float)

    global_nn = NearestNeighbors(n_neighbors=min(8, len(train_x)), metric="euclidean")
    global_nn.fit(train_x)
    global_d = global_nn.kneighbors(eval_x, return_distance=True)[0][:, -1]

    for c in classes:
        idx_eval = np.flatnonzero(pred == c)
        if idx_eval.size == 0:
            continue

        idx_train = np.flatnonzero(tr.labels == c)
        x_c = train_x[idx_train]
        n_c = len(x_c)
        k = min(8, n_c)

        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(x_c)
        d_eval = nn.kneighbors(eval_x[idx_eval], return_distance=True)[0][:, -1]

        if n_c > 1:
            leave_k = min(k + 1, n_c)
            leave_nn = NearestNeighbors(n_neighbors=leave_k, metric="euclidean")
            leave_nn.fit(x_c)
            d_train = leave_nn.kneighbors(x_c, return_distance=True)[0][:, -1]
            scale = np.median(d_train)
        else:
            scale = np.median(global_d) + eps

        local = np.log1p(d_eval / (scale + eps))
        relative = np.log1p(d_eval / (global_d[idx_eval] + eps))
        scores[idx_eval] = local + 0.25 * relative

    unseen = ~np.isin(pred, classes)
    if np.any(unseen):
        scores[unseen] = np.log1p(global_d[unseen] / (np.median(global_d) + eps))

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
