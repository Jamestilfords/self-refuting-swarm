import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Low-variance within-intent directions encode tightly constrained intent structure. "
    "A query should have close same-predicted-intent neighbours in this trailing subspace, while OOS queries tend to lack such local support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0055", "m_0053"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k_classes = tr.logits.shape[1]
    dim = x.shape[1]

    counts = np.bincount(y, minlength=k_classes).astype(float)
    sums = np.zeros((k_classes, dim), dtype=float)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(len(x) - k_classes, 1)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    tail_dim = min(48, dim)
    tail_vecs = vecs[:, :tail_dim]
    tail_vals = vals[:tail_dim]
    floor = max(float(np.max(vals)) * 1e-6, 1e-12)
    scale = 1.0 / np.sqrt(np.maximum(tail_vals, floor))

    train_tail = (centered @ tail_vecs) * scale
    pred_eval = np.argmax(ev.logits, axis=1)
    scores = np.empty(len(z), dtype=float)

    global_nn = NearestNeighbors(
        n_neighbors=min(10, len(train_tail)),
        metric="euclidean",
        algorithm="auto",
    ).fit(train_tail)

    for c in np.unique(pred_eval):
        query_idx = np.flatnonzero(pred_eval == c)
        query_tail = ((z[query_idx] - means[c]) @ tail_vecs) * scale
        class_idx = np.flatnonzero(y == c)

        if len(class_idx) >= 2:
            n_neighbors = min(10, len(class_idx))
            nn = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric="euclidean",
                algorithm="auto",
            ).fit(train_tail[class_idx])
            dist = nn.kneighbors(query_tail, return_distance=True)[0][:, -1]
        else:
            dist = global_nn.kneighbors(query_tail, return_distance=True)[0][:, -1]

        scores[query_idx] = dist

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
