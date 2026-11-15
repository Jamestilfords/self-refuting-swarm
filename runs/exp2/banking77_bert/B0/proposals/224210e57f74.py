import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A ViM-style residual measures feature energy outside the principal subspace of in-scope encoder representations. OOS queries often retain unusually large residual components that are not explained by the dominant training-feature manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0006"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_components = min(64, tr.pooled.shape[0] - 1, tr.pooled.shape[1])
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        iterated_power=2,
        random_state=314159,
        whiten=False,
    )
    z_train = pca.fit_transform(tr.pooled)
    z_eval = pca.transform(ev.pooled)

    train_centered = tr.pooled - pca.mean_
    eval_centered = ev.pooled - pca.mean_

    train_r2 = np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(z_train * z_train, axis=1),
        0.0,
    )
    eval_r2 = np.maximum(
        np.sum(eval_centered * eval_centered, axis=1)
        - np.sum(z_eval * z_eval, axis=1),
        0.0,
    )

    train_score = np.log1p(np.sqrt(train_r2))
    eval_score = np.log1p(np.sqrt(eval_r2))

    scale = np.quantile(train_score, 0.9) - np.quantile(train_score, 0.1)
    scale = max(float(scale), 1e-6)
    center = float(np.median(train_score))

    return (eval_score - center) / scale
