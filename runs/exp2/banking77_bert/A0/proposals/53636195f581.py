import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Features from in-scope queries should lie close to a principal subspace learned from training representations, whereas OOS queries tend to have larger residual components. Combining this train-calibrated residual with energy captures both representation-space and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[0], tr.pooled.shape[1])
    pca = PCA(n_components=max(1, dim), svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    def residual(x):
        centered = x - pca.mean_
        total_sq = np.sum(centered * centered, axis=1)
        projected = pca.transform(x)
        kept_sq = np.sum(projected * projected, axis=1)
        return np.sqrt(np.maximum(total_sq - kept_sq, 0.0))

    train_residual = residual(tr.pooled)
    eval_residual = residual(ev.pooled)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    def robust_scale(train_values, values):
        lo, hi = np.percentile(train_values, [10.0, 90.0])
        scale = hi - lo
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        center = np.median(train_values)
        return np.nan_to_num((values - center) / scale, nan=0.0, posinf=1e6, neginf=-1e6)

    residual_score = robust_scale(train_residual, eval_residual)
    energy_score = robust_scale(train_energy, eval_energy)
    return energy_score + residual_score
