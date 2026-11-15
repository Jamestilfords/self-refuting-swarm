import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A PCA residual measures encoder content outside the train-fitted in-scope manifold, "
    "while train-calibrated residual tail probability emphasizes unsupported directions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0147", "m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(192, tr.pooled.shape[1] - 1, tr.pooled.shape[0] - 1)
    pca = PCA(n_components=max(1, dim), svd_solver="randomized", random_state=2718)
    z_train = pca.fit_transform(tr.pooled)
    z_eval = pca.transform(ev.pooled)

    train_sq = np.sum(tr.pooled * tr.pooled, axis=1)
    train_proj_sq = np.sum(z_train * z_train, axis=1)
    train_residual = np.sqrt(np.maximum(train_sq - train_proj_sq, 0.0))

    eval_centered = ev.pooled - pca.mean_
    eval_sq = np.sum(eval_centered * eval_centered, axis=1)
    eval_proj_sq = np.sum(z_eval * z_eval, axis=1)
    eval_residual = np.sqrt(np.maximum(eval_sq - eval_proj_sq, 0.0))

    reference = np.sort(train_residual)
    ranks = np.searchsorted(reference, eval_residual, side="right")
    return ranks.astype(float) / (reference.size + 1.0)
