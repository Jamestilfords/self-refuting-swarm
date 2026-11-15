import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A ViM-style virtual OOS logit is formed from the norm outside a train-fitted principal feature subspace. In-scope examples should be explained by that subspace, while residual-heavy examples receive higher virtual-class posterior mass."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(192, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=314159)
    pca.fit(tr.pooled)

    def residual_norm(x):
        z = pca.transform(x)
        recon = z @ pca.components_ + pca.mean_
        return np.sqrt(np.sum((x - recon) ** 2, axis=1) + 1e-12)

    train_residual = residual_norm(tr.pooled)
    train_peak = np.max(tr.logits, axis=1)

    alpha = np.mean(np.abs(train_peak)) / (np.mean(train_residual) + 1e-12)
    alpha = float(np.clip(alpha, 1e-6, 1e6))

    virtual_logit = alpha * residual_norm(ev.pooled)
    known_log_norm = logsumexp(ev.logits, axis=1)
    return virtual_logit - np.logaddexp(known_log_norm, virtual_logit)
