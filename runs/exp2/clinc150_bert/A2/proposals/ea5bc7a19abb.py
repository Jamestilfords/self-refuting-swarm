import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A ViM-style residual norm captures encoder components unsupported by the dominant in-scope feature subspace, while energy captures classifier uncertainty. Training-calibrated addition combines semantic off-manifold evidence with weak intent evidence."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0069", "m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    rank = min(96, x_train.shape[0] - 1, x_train.shape[1])
    pca = PCA(n_components=rank, svd_solver="randomized", random_state=314159)
    pca.fit(x_train)

    mean = pca.mean_
    components = pca.components_

    train_centered = x_train - mean
    eval_centered = x_eval - mean

    train_proj = train_centered @ components.T
    eval_proj = eval_centered @ components.T

    train_residual_sq = np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(train_proj * train_proj, axis=1),
        0.0,
    )
    eval_residual_sq = np.maximum(
        np.sum(eval_centered * eval_centered, axis=1)
        - np.sum(eval_proj * eval_proj, axis=1),
        0.0,
    )

    train_residual = np.sqrt(train_residual_sq + 1e-12)
    eval_residual = np.sqrt(eval_residual_sq + 1e-12)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)

    energy_scale = np.std(train_energy) + 1e-12
    residual_scale = np.std(train_residual) + 1e-12
    alpha = energy_scale / residual_scale

    return eval_energy + alpha * eval_residual
