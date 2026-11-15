import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norms measure how far a representation lies outside the dominant in-scope feature subspace. Training-calibrated addition to energy combines off-manifold evidence with classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0129"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(256, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=17)
    pca.fit(tr.pooled)

    def residual_norm(x):
        centered = x - pca.mean_
        coords = centered @ pca.components_.T
        residual = centered - coords @ pca.components_
        return np.sqrt(np.sum(residual * residual, axis=1))

    train_residual = residual_norm(tr.pooled)
    eval_residual = residual_norm(ev.pooled)

    train_energy = ctx.h.energy(tr.logits)
    energy_scale = np.std(train_energy) + 1e-3
    residual_scale = np.std(train_residual) + 1e-6
    alpha = energy_scale / residual_scale

    scores = ctx.h.energy(ev.logits) + alpha * eval_residual
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
