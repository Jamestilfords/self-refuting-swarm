import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norm measures how far an encoder representation lies outside the dominant in-scope feature subspace. Adding a train-calibrated residual term to energy can expose OOS inputs with deceptively plausible classifier logits."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0131"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_components = min(64, tr.pooled.shape[1], tr.pooled.shape[0] - 1)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=17)
    train_proj = pca.fit_transform(tr.pooled)
    eval_proj = pca.transform(ev.pooled)

    train_centered_sq = np.sum((tr.pooled - pca.mean_) ** 2, axis=1)
    eval_centered_sq = np.sum((ev.pooled - pca.mean_) ** 2, axis=1)
    train_residual = np.sqrt(np.maximum(train_centered_sq - np.sum(train_proj ** 2, axis=1), 0.0))
    eval_residual = np.sqrt(np.maximum(eval_centered_sq - np.sum(eval_proj ** 2, axis=1), 0.0))

    train_energy = ctx.h.energy(tr.logits)
    energy_scale = np.percentile(train_energy, 75) - np.percentile(train_energy, 25)
    residual_scale = np.percentile(train_residual, 75) - np.percentile(train_residual, 25)
    weight = energy_scale / max(residual_scale, 1e-8)

    return ctx.h.energy(ev.logits) + weight * eval_residual
