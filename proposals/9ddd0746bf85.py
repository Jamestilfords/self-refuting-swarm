import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Trailing within-class covariance directions encode tightly constrained residual structure. "
    "OOS queries should have weak angular neighborhood support among examples of their predicted intent in this low-variance subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0220"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    pred = np.argmax(ev.logits, axis=1)
    n, d = x.shape
    k_classes = tr.logits.shape[1]

    global_mean = np.mean(x, axis=0)
    means = np.empty((k_classes, d), dtype=float)
    means[:] = global_mean
    for c in range(k_classes):
        mask = y == c
        if np.any(mask):
            means[c] = np.mean(x[mask], axis=0)

    residuals = x - means[np.clip(y, 0, k_classes - 1)]
    cov = (residuals.T @ residuals) / max(n - 1, 1)
    cov.flat[:: d + 1] += 1e-6 * (np.trace(cov) / max(d, 1) + 1e-12)

    vals, vecs = np.linalg.eigh(cov)
    trailing_dim = min(d, max(24, d // 8))
    idx = np.arange(trailing_dim)
    basis = vecs[:, idx]
    scale = np.sqrt(np.maximum(vals[idx], 1e-10))

    train_tail = (residuals @ basis) / scale
    eval_tail = (z - means[np.clip(pred, 0, k_classes - 1)]) @ basis
    eval_tail = eval_tail / scale

    def unit_rows(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)

    train_tail = unit_rows(train_tail)
    eval_tail = unit_rows(eval_tail)
    scores = np.empty(len(z), dtype=float)

    for c in np.unique(pred):
        query_mask = pred == c
        train_mask = y == c
        reference = train_tail[train_mask]
        if len(reference) < 2:
            reference = train_tail
        neighbors = min(10, len(reference))
        model = NearestNeighbors(n_neighbors=neighbors, metric="euclidean")
        model.fit(reference)
        distances, _ = model.kneighbors(eval_tail[query_mask])
        scores[query_mask] = distances[:, -1]

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
