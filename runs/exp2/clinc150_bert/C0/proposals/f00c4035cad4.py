import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norm measures how far an encoder representation lies outside the principal in-scope feature subspace. Combining this residual with energy catches both representation-level novelty and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    def residual_sq(x):
        centered = x - pca.mean_
        projected = pca.transform(x)
        total_sq = np.sum(centered * centered, axis=1)
        explained_sq = np.sum(projected * projected, axis=1)
        return np.maximum(total_sq - explained_sq, 0.0)

    train_residual = residual_sq(tr.pooled)
    eval_residual = residual_sq(ev.pooled)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    res_center = np.median(train_residual)
    res_scale = np.median(np.abs(train_residual - res_center)) * 1.4826
    energy_center = np.median(train_energy)
    energy_scale = np.median(np.abs(train_energy - energy_center)) * 1.4826

    res_scale = max(float(res_scale), 1e-8)
    energy_scale = max(float(energy_scale), 1e-8)

    residual_score = (eval_residual - res_center) / res_scale
    energy_score = (eval_energy - energy_center) / energy_scale
    return residual_score + energy_score
