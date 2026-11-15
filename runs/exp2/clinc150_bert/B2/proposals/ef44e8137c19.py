import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A ViM-style virtual-class margin compares residual representation energy outside the in-scope principal subspace against the classifier's centered winning-logit evidence. OOS inputs should have a larger virtual residual logit and weaker discriminative margin."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0127"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    n, d = x_train.shape
    n_comp = max(1, min(96, d, n - 1))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=314159)
    z_train = pca.fit_transform(x_train)
    z_eval = pca.transform(x_eval)

    train_centered = x_train - pca.mean_
    eval_centered = x_eval - pca.mean_

    train_residual_sq = np.sum(train_centered * train_centered, axis=1)
    train_residual_sq -= np.sum(z_train * z_train, axis=1)
    eval_residual_sq = np.sum(eval_centered * eval_centered, axis=1)
    eval_residual_sq -= np.sum(z_eval * z_eval, axis=1)

    r_train = np.sqrt(np.maximum(train_residual_sq, 0.0))
    r_eval = np.sqrt(np.maximum(eval_residual_sq, 0.0))

    train_logits = np.asarray(tr.logits, dtype=np.float64)
    eval_logits = np.asarray(ev.logits, dtype=np.float64)

    train_margin = np.max(train_logits, axis=1) - np.mean(train_logits, axis=1)
    eval_margin = np.max(eval_logits, axis=1) - np.mean(eval_logits, axis=1)

    alpha = np.mean(train_margin) / max(np.mean(r_train), 1e-8)
    alpha = max(float(alpha), 1e-8)

    scores = alpha * r_eval - eval_margin
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
