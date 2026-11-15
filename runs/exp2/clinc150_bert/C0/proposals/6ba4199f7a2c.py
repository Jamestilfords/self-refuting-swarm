import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norms measure feature mass outside the principal in-scope subspace, while an added virtual logit combines this geometric evidence with classifier confidence. OOS queries should produce larger residual-derived virtual logits than in-scope queries."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=17)
    pca.fit(tr.pooled)

    def residual_norm(x):
        z = x - pca.mean_
        proj = (z @ pca.components_.T) @ pca.components_
        return np.sqrt(np.maximum(np.sum((z - proj) ** 2, axis=1), 0.0))

    train_residual = residual_norm(tr.pooled)
    eval_residual = residual_norm(ev.pooled)

    denom = max(float(np.mean(train_residual)), 1e-8)
    scale = float(np.mean(np.abs(np.max(tr.logits, axis=1)))) / denom
    virtual_logit = scale * eval_residual

    return logsumexp(
        np.concatenate([ev.logits, virtual_logit[:, None]], axis=1),
        axis=1,
    )
