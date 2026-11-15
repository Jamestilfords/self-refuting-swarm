import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "After removing the intent centroid, in-scope encoder representations should lie near a "
    "shared low-dimensional within-intent manifold. OOS queries retain unusually large residual "
    "energy outside that learned manifold even when a classifier assigns them a known intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    centers = np.zeros((n_classes, dim), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    np.add.at(centers, labels, x_train)
    centers /= np.maximum(counts[:, None], 1.0)

    train_centered = x_train - centers[labels]
    n_components = min(96, dim, max(1, train_centered.shape[0] - 1))
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=1729,
    )
    pca.fit(train_centered)

    predicted = np.argmax(ev.logits, axis=1)
    eval_centered = x_eval - centers[predicted]
    projected = pca.transform(eval_centered)
    reconstruction = projected @ pca.components_ + pca.mean_
    residual = eval_centered - reconstruction

    scores = np.sqrt(np.maximum(np.sum(residual * residual, axis=1), 0.0))
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
