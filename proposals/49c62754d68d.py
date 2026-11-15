import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A posterior-weighted intent prototype removes the representation component explained by the "
    "classifier's likely known intents, while the remaining norm outside a train-fitted within-intent "
    "principal subspace measures unsupported encoder variation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004", "m_0067"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    n_classes = tr.logits.shape[1]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    means = np.zeros((n_classes, x_train.shape[1]), dtype=np.float64)
    np.add.at(means, labels, x_train)
    means /= counts[:, None]

    train_centered = x_train - means[labels]
    n_components = min(256, train_centered.shape[0] - 1, train_centered.shape[1] - 1)
    n_components = max(1, n_components)

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=271828,
        whiten=False,
    )
    pca.fit(train_centered)

    def residual_norm(z):
        coords = (z - pca.mean_) @ pca.components_.T
        recon = coords @ pca.components_ + pca.mean_
        return np.sqrt(np.sum((z - recon) ** 2, axis=1))

    train_norm = residual_norm(train_centered)
    scale = np.quantile(train_norm, 0.90)
    scale = max(float(scale), 1e-8)

    probs = ctx.h.softmax(ev.logits)
    soft_prototype = probs @ means
    eval_centered = x_eval - soft_prototype
    scores = residual_norm(eval_centered) / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
