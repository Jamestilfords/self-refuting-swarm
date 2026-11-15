import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A ViM-style residual isolates feature components unsupported by the in-scope principal subspace. "
    "Aggregating coordinate-wise residual exceedances beyond train-fitted high quantiles emphasizes broad novel semantic content without relying only on residual L2 magnitude."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0147", "m_0042"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    n, d = x_train.shape
    n_comp = max(1, min(256, d - 1, n - 1))
    pca = PCA(
        n_components=n_comp,
        svd_solver="randomized",
        random_state=173,
        iterated_power=4,
    )
    pca.fit(x_train)

    train_proj = pca.inverse_transform(pca.transform(x_train))
    eval_proj = pca.inverse_transform(pca.transform(x_eval))
    train_residual = x_train - train_proj
    eval_residual = x_eval - eval_proj

    scale = np.quantile(np.abs(train_residual), 0.95, axis=0)
    fallback = np.sqrt(np.mean(train_residual * train_residual, axis=0))
    scale = np.maximum(scale, np.maximum(fallback, 1e-8))

    normalized = np.abs(eval_residual) / scale
    exceedance = np.maximum(normalized - 1.0, 0.0)
    sparse_burden = np.mean(exceedance * exceedance, axis=1)
    residual_energy = np.mean(normalized * normalized, axis=1)

    return sparse_burden + 1e-4 * residual_energy
