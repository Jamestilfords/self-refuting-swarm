import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "ViM-style residual norm measures representation energy outside the principal "
    "in-scope feature subspace, while energy captures classifier uncertainty. "
    "Training-calibrated combination makes either unsupported geometry or diffuse logits OOS-like."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0130"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(n_components=64, svd_solver="randomized", random_state=17)
    pca.fit(tr.pooled)

    tr_centered = tr.pooled - pca.mean_
    ev_centered = ev.pooled - pca.mean_
    tr_proj = tr_centered @ pca.components_.T
    ev_proj = ev_centered @ pca.components_.T

    tr_residual = np.sqrt(
        np.maximum(
            np.sum(tr_centered * tr_centered, axis=1) - np.sum(tr_proj * tr_proj, axis=1),
            0.0,
        )
    )
    ev_residual = np.sqrt(
        np.maximum(
            np.sum(ev_centered * ev_centered, axis=1) - np.sum(ev_proj * ev_proj, axis=1),
            0.0,
        )
    )

    tr_energy = ctx.h.energy(tr.logits)
    ev_energy = ctx.h.energy(ev.logits)

    def robust_standardize(values, train_values):
        med = np.median(train_values)
        scale = np.median(np.abs(train_values - med))
        scale = max(float(scale), 1e-6)
        return np.clip((values - med) / scale, -50.0, 50.0)

    residual_score = robust_standardize(ev_residual, tr_residual)
    energy_score = robust_standardize(ev_energy, tr_energy)
    return residual_score + energy_score
