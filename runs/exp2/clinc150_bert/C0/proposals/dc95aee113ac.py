import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norm measures encoder variation outside the principal in-scope feature subspace, while energy retains classifier uncertainty information. Robust train-fitted scaling makes the two signals comparable without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_comp = min(128, tr.pooled.shape[0], tr.pooled.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=17)
    z_train = pca.fit_transform(tr.pooled)
    z_eval = pca.transform(ev.pooled)

    train_centered_sq = np.sum((tr.pooled - pca.mean_) ** 2, axis=1)
    eval_centered_sq = np.sum((ev.pooled - pca.mean_) ** 2, axis=1)
    train_residual = np.maximum(train_centered_sq - np.sum(z_train ** 2, axis=1), 0.0)
    eval_residual = np.maximum(eval_centered_sq - np.sum(z_eval ** 2, axis=1), 0.0)

    train_residual = np.log1p(train_residual)
    eval_residual = np.log1p(eval_residual)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    def robust_scale(x, ref):
        med = np.median(ref)
        scale = np.median(np.abs(ref - med))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(ref)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return (x - med) / scale

    return robust_scale(eval_energy, train_energy) + robust_scale(
        eval_residual, train_residual
    )
