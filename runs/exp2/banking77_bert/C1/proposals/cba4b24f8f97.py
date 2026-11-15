import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norms measure encoder variation outside the principal in-scope feature subspace. Calibrating that residual against the maximum class logit yields high scores when an item is both representation-novel and weakly supported by the classifier."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_comp = min(128, tr.pooled.shape[0] - 1, tr.pooled.shape[1] - 1)
    n_comp = max(n_comp, 1)

    pca = PCA(
        n_components=n_comp,
        svd_solver="randomized",
        random_state=2718,
        whiten=False,
    )
    train_proj = pca.fit_transform(tr.pooled)
    eval_proj = pca.transform(ev.pooled)

    train_centered = tr.pooled - pca.mean_
    eval_centered = ev.pooled - pca.mean_

    train_residual = np.sqrt(np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(train_proj * train_proj, axis=1),
        0.0,
    ))
    eval_residual = np.sqrt(np.maximum(
        np.sum(eval_centered * eval_centered, axis=1)
        - np.sum(eval_proj * eval_proj, axis=1),
        0.0,
    ))

    train_maxlogit = np.max(tr.logits, axis=1)
    residual_mean = max(float(np.mean(train_residual)), 1e-8)
    alpha = float(np.mean(train_maxlogit)) / residual_mean

    scores = alpha * eval_residual - np.max(ev.logits, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
