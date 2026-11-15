import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode constraints that in-scope examples of an "
    "intent should satisfy. Class-conditional neighbor distance in this whitened trailing "
    "subspace measures local violations of those constraints."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0051", "m_0073"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes, inverse = np.unique(y, return_inverse=True)
    n, d = x.shape

    means = np.zeros((len(classes), d), dtype=float)
    np.add.at(means, inverse, x)
    counts = np.bincount(inverse, minlength=len(classes)).astype(float)
    means /= np.maximum(counts[:, None], 1.0)

    residuals = x - means[inverse]
    denom = max(n - len(classes), 1)
    cov = (residuals.T @ residuals) / float(denom)
    eigvals, eigvecs = np.linalg.eigh(cov)

    tail_dim = min(96, max(16, d // 4))
    tail_vals = eigvals[:tail_dim]
    tail_vecs = eigvecs[:, :tail_dim]

    positive = eigvals[eigvals > 0.0]
    floor = (np.median(positive) if positive.size else 1.0) * 1e-3
    scales = np.sqrt(np.maximum(tail_vals, max(floor, 1e-12)))

    train_tail = (x @ tail_vecs) / scales
    eval_tail = (z @ tail_vecs) / scales
    predicted = np.argmax(ev.logits, axis=1)

    global_k = min(5, n)
    global_nn = NearestNeighbors(n_neighbors=global_k, metric="euclidean")
    global_nn.fit(train_tail)
    scores = global_nn.kneighbors(eval_tail, return_distance=True)[0][:, -1]

    for class_index, label in enumerate(classes):
        query_mask = predicted == label
        if not np.any(query_mask):
            continue
        train_mask = inverse == class_index
        class_count = int(np.sum(train_mask))
        if class_count == 0:
            continue
        k = min(5, class_count)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(train_tail[train_mask])
        scores[query_mask] = nn.kneighbors(
            eval_tail[query_mask], return_distance=True
        )[0][:, -1]

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
