import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "The fraction of feature energy outside a train-derived principal semantic subspace measures manifold violation while being less sensitive to overall feature norm. Combining its train-standardized value with energy captures both unsupported semantic variation and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0109"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = np.mean(x_train, axis=0, keepdims=True)
    train_centered = x_train - mean
    eval_centered = x_eval - mean

    n_components = min(128, train_centered.shape[1], max(1, train_centered.shape[0] - 1))
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=173,
        whiten=False,
    )
    pca.fit(train_centered)

    components = pca.components_
    train_proj = train_centered @ components.T
    eval_proj = eval_centered @ components.T

    train_total = np.sum(train_centered * train_centered, axis=1)
    eval_total = np.sum(eval_centered * eval_centered, axis=1)
    train_resid = np.maximum(train_total - np.sum(train_proj * train_proj, axis=1), 0.0)
    eval_resid = np.maximum(eval_total - np.sum(eval_proj * eval_proj, axis=1), 0.0)

    scale = np.median(train_total) + 1e-12
    train_fraction = train_resid / (train_total + scale)
    eval_fraction = eval_resid / (eval_total + scale)

    resid_mu = np.mean(train_fraction)
    resid_sd = np.std(train_fraction) + 1e-12
    resid_z = (eval_fraction - resid_mu) / resid_sd

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)
    energy_mu = np.mean(train_energy)
    energy_sd = np.std(train_energy) + 1e-12
    energy_z = (eval_energy - energy_mu) / energy_sd

    scores = energy_z + resid_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
