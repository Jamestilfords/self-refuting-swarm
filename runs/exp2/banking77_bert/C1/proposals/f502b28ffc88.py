import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "OOS queries may be assigned a plausible intent class by the logits but retain a large component outside the shared in-scope principal subspace relative to that predicted class's centroid. This class-conditioned residual differs from a global ViM norm by measuring mismatch to the specific intent the classifier selected."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    n_classes = tr.logits.shape[1]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    sums = np.zeros((n_classes, x_train.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, x_train)
    centroids = sums / np.maximum(counts[:, None], 1.0)

    n_comp = min(160, x_train.shape[0] - 1, x_train.shape[1])
    if n_comp < 1:
        return np.sum((x_eval - np.mean(x_train, axis=0)) ** 2, axis=1)

    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=2718)
    pca.fit(x_train)

    pred = np.argmax(ev.logits, axis=1)
    delta = x_eval - centroids[pred]
    projected = delta @ pca.components_.T
    residual_sq = np.sum(delta * delta, axis=1) - np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(residual_sq, 0.0)

    train_delta = x_train - centroids[labels]
    train_projected = train_delta @ pca.components_.T
    train_residual_sq = np.maximum(
        np.sum(train_delta * train_delta, axis=1)
        - np.sum(train_projected * train_projected, axis=1),
        0.0,
    )
    scale = np.median(train_residual_sq) + 1e-8

    return np.log1p(residual_sq / scale)
