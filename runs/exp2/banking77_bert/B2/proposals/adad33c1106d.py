import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norm measures feature content outside the principal in-scope representation subspace, then inserts that residual as a virtual OOS logit. This flags samples that are both poorly represented by known feature structure and not strongly supported by classifier logits."
SOURCE = "https://github.com/haoqiwang/vim/"
PARENTS = []
MEMORY_USED = ["m_0056", "m_0067", "m_0068"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    n_components = min(128, x_train.shape[1] - 1, x_train.shape[0] - 1)
    n_components = max(1, n_components)

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=2027,
        whiten=False,
    )
    pca.fit(x_train)

    centered_train = x_train - pca.mean_
    centered_eval = x_eval - pca.mean_
    train_proj = pca.transform(x_train)
    eval_proj = pca.transform(x_eval)

    train_residual_sq = np.maximum(
        np.sum(centered_train * centered_train, axis=1)
        - np.sum(train_proj * train_proj, axis=1),
        0.0,
    )
    eval_residual_sq = np.maximum(
        np.sum(centered_eval * centered_eval, axis=1)
        - np.sum(eval_proj * eval_proj, axis=1),
        0.0,
    )
    train_residual = np.sqrt(train_residual_sq)
    eval_residual = np.sqrt(eval_residual_sq)

    mean_residual = max(float(np.mean(train_residual)), 1e-12)
    mean_max_logit = float(np.mean(np.max(tr.logits, axis=1)))
    alpha = max(mean_max_logit / mean_residual, 1e-6)

    virtual_logit = alpha * eval_residual
    logsumexp_logits = np.logaddexp.reduce(ev.logits, axis=1)
    return -np.logaddexp(logsumexp_logits, virtual_logit)
