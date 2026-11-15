import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Rather than using residual magnitude alone, this measures the fraction of a feature's "
    "centered energy lying outside the train-fitted principal in-scope subspace. OOS queries "
    "should often point away from the known manifold even when their overall feature norm is ordinary."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0147"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(
        n_components=dim,
        svd_solver="randomized",
        random_state=1729,
        whiten=False,
    )
    pca.fit(tr.pooled)

    train_centered = tr.pooled - pca.mean_
    eval_centered = ev.pooled - pca.mean_
    train_proj = pca.transform(tr.pooled)
    eval_proj = pca.transform(ev.pooled)

    eps = 1e-12
    train_total = np.sum(train_centered * train_centered, axis=1)
    eval_total = np.sum(eval_centered * eval_centered, axis=1)
    train_kept = np.sum(train_proj * train_proj, axis=1)
    eval_kept = np.sum(eval_proj * eval_proj, axis=1)

    train_fraction = np.clip((train_total - train_kept) / (train_total + eps), 0.0, 1.0)
    eval_fraction = np.clip((eval_total - eval_kept) / (eval_total + eps), 0.0, 1.0)

    median = np.median(train_fraction)
    mad = np.median(np.abs(train_fraction - median))
    scale = max(1.4826 * mad, 1e-6)

    return (eval_fraction - median) / scale
