import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A ViM-inspired residual fraction measures the proportion of encoder feature energy outside the dominant in-scope subspace, rather than raw residual magnitude alone. Adding calibrated energy combines representation novelty with classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_components = min(96, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=2718)
    train_proj = pca.fit_transform(tr.pooled)
    eval_proj = pca.transform(ev.pooled)

    train_centered = tr.pooled - pca.mean_
    eval_centered = ev.pooled - pca.mean_

    train_total_sq = np.sum(train_centered * train_centered, axis=1)
    eval_total_sq = np.sum(eval_centered * eval_centered, axis=1)
    train_kept_sq = np.sum(train_proj * train_proj, axis=1)
    eval_kept_sq = np.sum(eval_proj * eval_proj, axis=1)

    train_residual_fraction = np.sqrt(
        np.maximum(train_total_sq - train_kept_sq, 0.0) / (train_total_sq + 1e-12)
    )
    eval_residual_fraction = np.sqrt(
        np.maximum(eval_total_sq - eval_kept_sq, 0.0) / (eval_total_sq + 1e-12)
    )

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    residual_med = np.median(train_residual_fraction)
    residual_scale = np.median(np.abs(train_residual_fraction - residual_med)) + 1e-8
    energy_med = np.median(train_energy)
    energy_scale = np.median(np.abs(train_energy - energy_med)) + 1e-8

    residual_score = (eval_residual_fraction - residual_med) / residual_scale
    energy_score = (eval_energy - energy_med) / energy_scale

    return residual_score + 0.35 * energy_score
