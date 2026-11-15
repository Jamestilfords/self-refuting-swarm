import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "The residual norm outside a principal subspace fitted on in-scope pooled features measures semantic components unsupported by the known-intent representation manifold. Adding train-standardized energy also captures logit uncertainty when an input remains close to that manifold."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0147"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    n_comp = min(256, x_train.shape[1], max(1, x_train.shape[0] - 1))
    pca = PCA(
        n_components=n_comp,
        svd_solver="randomized",
        iterated_power=3,
        random_state=1729,
    )
    pca.fit(x_train)

    mean = pca.mean_
    components = pca.components_
    train_centered = x_train - mean
    train_total_sq = np.sum(train_centered * train_centered, axis=1)
    train_scale = max(float(np.median(train_total_sq)), 1e-8)

    def residual_fraction(x):
        centered = x - mean
        total_sq = np.sum(centered * centered, axis=1)
        projected = centered @ components.T
        explained_sq = np.sum(projected * projected, axis=1)
        residual_sq = np.maximum(total_sq - explained_sq, 0.0)
        return residual_sq / (total_sq + train_scale)

    train_residual = residual_fraction(x_train)
    eval_residual = residual_fraction(x_eval)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)

    residual_mu = float(np.mean(train_residual))
    residual_sd = max(float(np.std(train_residual)), 1e-6)
    energy_mu = float(np.mean(train_energy))
    energy_sd = max(float(np.std(train_energy)), 1e-6)

    residual_z = (eval_residual - residual_mu) / residual_sd
    energy_z = (eval_energy - energy_mu) / energy_sd
    return residual_z + energy_z
