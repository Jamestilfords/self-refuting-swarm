import numpy as np
from sklearn.neighbors import NearestNeighbors

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Low-variance within-intent directions can encode characteristic signed residual patterns, "
    "so OOS queries should lack nearby in-scope trailing-eigenspace sign codes for their predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0087"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    n, d = x.shape
    k_classes = tr.logits.shape[1]

    counts = np.bincount(y, minlength=k_classes).astype(float)
    sums = np.zeros((k_classes, d), dtype=float)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    denom = max(n - np.count_nonzero(counts), 1)
    cov = (centered.T @ centered) / float(denom)
    ridge = 1e-6 * (np.trace(cov) / max(d, 1) + 1.0)
    eigvals, eigvecs = np.linalg.eigh(cov + ridge * np.eye(d))

    q = min(48, d)
    trailing = eigvecs[:, :q]

    train_code = ((centered @ trailing) >= 0.0).astype(np.uint8)
    pred = np.argmax(ev.logits, axis=1)
    eval_centered = z - means[pred]
    eval_code = ((eval_centered @ trailing) >= 0.0).astype(np.uint8)

    scores = np.empty(z.shape[0], dtype=float)
    global_model = NearestNeighbors(
        n_neighbors=min(5, n), metric="hamming", algorithm="brute"
    ).fit(train_code)

    for c in np.unique(pred):
        rows = np.flatnonzero(pred == c)
        members = np.flatnonzero(y == c)

        if members.size >= 2:
            model = NearestNeighbors(
                n_neighbors=min(5, members.size),
                metric="hamming",
                algorithm="brute",
            ).fit(train_code[members])
        else:
            model = global_model

        distances, _ = model.kneighbors(eval_code[rows])
        scores[rows] = distances.mean(axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
