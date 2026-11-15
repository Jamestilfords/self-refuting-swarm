import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions capture tightly constrained intent-specific "
    "variation. A query unsupported by nearby examples of its predicted intent in this "
    "low-variance residual space is likely out of scope."
)
SOURCE = "https://deel-ai.github.io/oodeel/latest/notebooks/torch/demo_vim_torch/"
PARENTS = []
MEMORY_USED = ["m_0060", "m_0062", "m_0068"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n_features = x_train.shape[1]
    n_classes = int(max(np.max(y), np.argmax(tr.logits, axis=1).max()) + 1)

    means = np.zeros((n_classes, n_features), dtype=np.float64)
    counts = np.bincount(y, minlength=n_classes).astype(np.int64)
    for c in range(n_classes):
        if counts[c] > 0:
            means[c] = x_train[y == c].mean(axis=0)

    residuals = x_train - means[y]
    cov = (residuals.T @ residuals) / max(len(x_train) - n_classes, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n_trailing = min(32, n_features)
    vals = np.maximum(eigvals[:n_trailing], 1e-8)
    basis = eigvecs[:, :n_trailing] / np.sqrt(vals)[None, :]

    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    scores = np.empty(len(x_eval), dtype=np.float64)

    for c in np.unique(pred_eval):
        query_idx = np.flatnonzero(pred_eval == c)
        support_idx = np.flatnonzero(y == c)
        if support_idx.size == 0:
            support_idx = np.flatnonzero(pred_train == c)
        if support_idx.size == 0:
            support_idx = np.arange(len(x_train))

        support = (x_train[support_idx] - means[c]) @ basis
        queries = (x_eval[query_idx] - means[c]) @ basis
        k = min(5, support.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        nn.fit(support)
        distances, _ = nn.kneighbors(queries)
        scores[query_idx] = distances[:, -1]

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
