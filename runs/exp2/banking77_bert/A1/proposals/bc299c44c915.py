import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "After removing the predicted intent's training mean, in-scope queries should lie near "
    "shared within-intent principal variation, while OOS queries retain a larger orthogonal residual."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    sums = np.zeros((n_classes, dim), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    np.add.at(sums, labels, x_train)

    global_mean = x_train.mean(axis=0)
    means = np.empty_like(sums)
    present = counts > 0
    means[present] = sums[present] / counts[present, None]
    means[~present] = global_mean

    train_centered = x_train - means[labels]
    n_components = min(128, train_centered.shape[0] - 1, dim)
    pca = PCA(
        n_components=max(1, n_components),
        svd_solver="randomized",
        random_state=271828,
    )
    pca.fit(train_centered)

    predicted = np.argmax(ev.logits, axis=1)
    eval_centered = x_eval - means[predicted]
    shifted = eval_centered - pca.mean_
    projected = pca.transform(eval_centered)

    total_sq = np.einsum("ij,ij->i", shifted, shifted)
    explained_sq = np.einsum("ij,ij->i", projected, projected)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)
    return np.sqrt(residual_sq)
