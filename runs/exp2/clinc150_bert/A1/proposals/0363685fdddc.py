import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norm measures how much an encoder representation lies outside the principal in-scope feature subspace. Combining this train-calibrated residual with energy captures both representation-space novelty and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    dim = min(128, x_train.shape[0], x_train.shape[1])
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=137)
    pca.fit(x_train)

    centered_train = x_train - pca.mean_
    centered_eval = x_eval - pca.mean_
    basis = pca.components_

    train_total = np.sum(centered_train * centered_train, axis=1)
    train_proj = np.sum((centered_train @ basis.T) ** 2, axis=1)
    eval_total = np.sum(centered_eval * centered_eval, axis=1)
    eval_proj = np.sum((centered_eval @ basis.T) ** 2, axis=1)

    residual_train = np.sqrt(np.maximum(train_total - train_proj, 0.0))
    residual_eval = np.sqrt(np.maximum(eval_total - eval_proj, 0.0))

    energy_train = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    energy_eval = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)

    def robust_standardize(values, reference):
        med = np.median(reference)
        scale = np.median(np.abs(reference - med))
        scale = max(float(scale), 1e-8)
        return (values - med) / scale

    residual_score = robust_standardize(residual_eval, residual_train)
    energy_score = robust_standardize(energy_eval, energy_train)
    return np.nan_to_num(residual_score + energy_score, nan=0.0, posinf=1e12, neginf=-1e12)
