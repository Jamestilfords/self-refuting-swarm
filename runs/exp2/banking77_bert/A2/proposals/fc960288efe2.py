import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Feature mass outside the dominant in-scope principal subspace is a ViM-style novelty signal, since in-scope encoder outputs should concentrate near that subspace. Train-calibrated residual magnitude is added to energy so classifier uncertainty and representation-space novelty reinforce one another."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0093"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(192, tr.pooled.shape[1], tr.pooled.shape[0] - 1)
    pca = PCA(n_components=max(1, dim), svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    def residual_norm(x):
        centered = x - pca.mean_
        projected = centered @ pca.components_.T
        total_sq = np.sum(centered * centered, axis=1)
        kept_sq = np.sum(projected * projected, axis=1)
        return np.sqrt(np.maximum(total_sq - kept_sq, 0.0))

    train_residual = residual_norm(tr.pooled)
    eval_residual = residual_norm(ev.pooled)

    med_r = np.median(train_residual)
    scale_r = np.median(np.abs(train_residual - med_r)) * 1.4826 + 1e-6

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    med_e = np.median(train_energy)
    scale_e = np.median(np.abs(train_energy - med_e)) * 1.4826 + 1e-6

    residual_score = (eval_residual - med_r) / scale_r
    energy_score = (eval_energy - med_e) / scale_e
    return energy_score + residual_score
