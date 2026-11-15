import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "This score measures residual norm after removing a class-conditioned principal subspace learned from within-class bottleneck variations. OOS queries should be poorly reconstructed even when they are close to a predicted class mean."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    sums = np.zeros((n_classes, dim), dtype=np.float64)
    np.add.at(sums, labels, x_train)
    means = sums / np.maximum(counts[:, None], 1.0)

    train_centered = x_train - means[labels]
    n_components = min(64, dim, max(1, train_centered.shape[0] - 1))
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=2718,
    )
    pca.fit(train_centered)

    predicted = np.argmax(ev.logits, axis=1)
    eval_centered = x_eval - means[predicted]

    coordinates = eval_centered @ pca.components_.T
    reconstructed = coordinates @ pca.components_
    residual = eval_centered - reconstructed
    return np.sqrt(np.maximum(np.sum(residual * residual, axis=1), 0.0))
