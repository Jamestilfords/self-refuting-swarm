import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Queries outside the training feature manifold should have a larger component orthogonal to the principal training subspace. Combining this ViM-style residual with energy captures both representation-space novelty and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)

    n_comp = min(128, x_train.shape[0], x_train.shape[1])
    pca = PCA(
        n_components=n_comp,
        svd_solver="randomized",
        iterated_power=3,
        random_state=2718,
    )
    pca.fit(x_train)

    def residual_norm(x):
        centered = x - pca.mean_
        total_sq = np.sum(centered * centered, axis=1)
        coords = centered @ pca.components_.T
        projected_sq = np.sum(coords * coords, axis=1)
        return np.sqrt(np.maximum(total_sq - projected_sq, 0.0))

    train_residual = residual_norm(x_train)
    eval_residual = residual_norm(x_eval)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    res_center = np.median(train_residual)
    energy_center = np.median(train_energy)
    res_scale = np.median(np.abs(train_residual - res_center))
    energy_scale = np.median(np.abs(train_energy - energy_center))

    if not np.isfinite(res_scale) or res_scale < 1e-8:
        res_scale = 1.0
    if not np.isfinite(energy_scale) or energy_scale < 1e-8:
        energy_scale = 1.0

    scores = (eval_residual - res_center) / res_scale
    scores += (eval_energy - energy_center) / energy_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
