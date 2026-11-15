import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Unsupported pooled-feature residuals are most informative when they are unusually large for the "
    "query's classifier-energy regime, rather than merely large in absolute terms."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0015", "m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_train, dim = tr.pooled.shape
    n_comp = min(256, dim, max(1, n_train - 1))

    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=271)
    pca.fit(tr.pooled)

    mean = pca.mean_
    components = pca.components_

    def residual_sq(x):
        centered = x - mean
        total = np.sum(centered * centered, axis=1)
        projected = centered @ components.T
        explained = np.sum(projected * projected, axis=1)
        return np.maximum(total - explained, 0.0)

    train_res = np.sqrt(residual_sq(tr.pooled))
    eval_res = np.sqrt(residual_sq(ev.pooled))

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    n_bins = min(16, max(4, int(np.sqrt(n_train))))
    edges = np.quantile(train_energy, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.maximum.accumulate(edges)

    train_bin = np.searchsorted(edges[1:-1], train_energy, side="right")
    eval_bin = np.searchsorted(edges[1:-1], eval_energy, side="right")

    global_med = np.median(train_res)
    global_mad = np.median(np.abs(train_res - global_med))
    global_scale = max(1.4826 * global_mad, 1e-8)

    medians = np.empty(n_bins, dtype=float)
    scales = np.empty(n_bins, dtype=float)
    for b in range(n_bins):
        values = train_res[train_bin == b]
        if values.size < 12:
            medians[b] = global_med
            scales[b] = global_scale
        else:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[b] = med
            scales[b] = max(1.4826 * mad, 0.1 * global_scale, 1e-8)

    return (eval_res - medians[eval_bin]) / scales[eval_bin]
