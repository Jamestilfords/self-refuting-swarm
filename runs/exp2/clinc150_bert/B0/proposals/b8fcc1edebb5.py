import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "ViM-style residual energy measures how much a pooled representation lies outside the dominant in-scope feature subspace. Combining this train-calibrated residual with energy highlights queries that are both geometrically atypical and classifier-uncertain."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    n_comp = max(1, min(128, x_train.shape[0] - 1, x_train.shape[1]))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=137)
    pca.fit(x_train)

    centered = x_eval - pca.mean_
    total_sq = np.sum(centered * centered, axis=1)
    projected = centered @ pca.components_.T
    explained_sq = np.sum(projected * projected, axis=1)
    residual = np.sqrt(np.maximum(total_sq - explained_sq, 0.0))

    train_centered = x_train - pca.mean_
    train_projected = train_centered @ pca.components_.T
    train_residual = np.sqrt(
        np.maximum(
            np.sum(train_centered * train_centered, axis=1)
            - np.sum(train_projected * train_projected, axis=1),
            0.0,
        )
    )

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)

    def robust_scale(values):
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = q75 - q25
        if not np.isfinite(scale) or scale < 1e-12:
            scale = np.std(values)
        return scale if np.isfinite(scale) and scale >= 1e-12 else 1.0

    residual_z = (residual - np.median(train_residual)) / robust_scale(train_residual)
    energy_z = (eval_energy - np.median(train_energy)) / robust_scale(train_energy)
    return np.asarray(residual_z + 0.5 * energy_z, dtype=np.float64)
