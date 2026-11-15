import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Out-of-scope inputs tend to have a larger encoder-feature residual outside the principal in-scope subspace; combining this residual with energy captures complementary representation and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_comp = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=137)
    pca.fit(tr.pooled)

    def residual_norm(x):
        centered = x - pca.mean_
        projection = centered @ pca.components_.T
        residual_sq = np.sum(centered * centered, axis=1) - np.sum(projection * projection, axis=1)
        return np.sqrt(np.maximum(residual_sq, 0.0))

    train_residual = residual_norm(tr.pooled)
    eval_residual = residual_norm(ev.pooled)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    res_center = np.median(train_residual)
    ene_center = np.median(train_energy)
    res_scale = np.median(np.abs(train_residual - res_center)) * 1.4826 + 1e-8
    ene_scale = np.median(np.abs(train_energy - ene_center)) * 1.4826 + 1e-8

    return (eval_energy - ene_center) / ene_scale + (eval_residual - res_center) / res_scale
