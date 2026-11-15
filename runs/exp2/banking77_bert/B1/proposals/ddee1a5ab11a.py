import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions encode semantic constraints that are weakly represented by dominant intent variation. "
    "A query is OOS-like when it lacks locally adaptive same-intent support in this low-variance residual geometry."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0110", "m_0114"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    classes, inverse = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    dim = x.shape[1]

    means = np.zeros((n_classes, dim), dtype=np.float64)
    np.add.at(means, inverse, x)
    counts = np.bincount(inverse, minlength=n_classes).astype(np.float64)
    means /= np.maximum(counts[:, None], 1.0)

    residuals = x - means[inverse]
    cov = residuals.T @ residuals / max(len(x) - n_classes, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    tail_dim = min(32, dim)
    tail_vals = np.maximum(eigvals[:tail_dim], 0.0)
    positive = eigvals[eigvals > 1e-12]
    floor = np.median(positive) * 1e-4 if positive.size else 1e-6
    scale = 1.0 / np.sqrt(tail_vals + max(floor, 1e-8))
    basis = eigvecs[:, :tail_dim] * scale[None, :]

    tx = residuals @ basis
    pred_labels = np.argmax(ev.logits, axis=1)
    class_to_pos = {int(c): i for i, c in enumerate(classes)}
    pred_pos = np.array(
        [class_to_pos.get(int(c), -1) for c in pred_labels], dtype=np.int64
    )

    scores = np.empty(len(z), dtype=np.float64)
    global_scale = np.sqrt(np.mean(np.sum(tx * tx, axis=1)) + 1e-8)

    for ci in range(n_classes):
        eval_idx = np.flatnonzero(pred_pos == ci)
        if eval_idx.size == 0:
            continue

        train_idx = np.flatnonzero(inverse == ci)
        local_train = tx[train_idx]
        q = (z[eval_idx] - means[ci]) @ basis
        n_local = len(local_train)

        if n_local < 3:
            d = np.sqrt(np.sum((q - np.mean(local_train, axis=0)) ** 2, axis=1))
            scores[eval_idx] = d / global_scale
            continue

        k = min(8, n_local - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(local_train)
        train_dist = nn.kneighbors(local_train, return_distance=True)[0][:, -1]
        query_dist, query_neighbors = nn.kneighbors(q, n_neighbors=k, return_distance=True)
        radius = query_dist[:, -1]
        neighbor_scales = train_dist[query_neighbors]
        adaptive_scale = np.exp(
            np.mean(np.log(np.maximum(neighbor_scales, 1e-8)), axis=1)
        )
        scores[eval_idx] = np.log1p(radius / np.maximum(adaptive_scale, 1e-8))

    unknown = pred_pos < 0
    if np.any(unknown):
        q = z[unknown] @ basis
        nn = NearestNeighbors(n_neighbors=min(8, len(tx)), metric="euclidean").fit(tx)
        scores[unknown] = np.log1p(nn.kneighbors(q, return_distance=True)[0][:, -1])

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
