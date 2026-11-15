import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "In-scope encoder representations should lie near a low-dimensional principal subspace learned from training data, while OOS queries often retain larger residual components. Combining this residual with energy captures both representation-level novelty and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_comp = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=2027)
    pca.fit(tr.pooled)

    train_proj = pca.transform(tr.pooled)
    eval_proj = pca.transform(ev.pooled)

    train_centered_sq = np.sum((tr.pooled - pca.mean_) ** 2, axis=1)
    eval_centered_sq = np.sum((ev.pooled - pca.mean_) ** 2, axis=1)
    train_residual = np.maximum(
        train_centered_sq - np.sum(train_proj ** 2, axis=1), 0.0
    )
    eval_residual = np.maximum(
        eval_centered_sq - np.sum(eval_proj ** 2, axis=1), 0.0
    )

    res_center = np.median(train_residual)
    res_scale = np.median(np.abs(train_residual - res_center))
    res_scale = max(float(res_scale), 1e-8)
    residual_score = (eval_residual - res_center) / res_scale

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    energy_center = float(np.mean(train_energy))
    energy_scale = max(float(np.std(train_energy)), 1e-8)
    energy_score = (eval_energy - energy_center) / energy_scale

    return np.nan_to_num(
        residual_score + energy_score,
        nan=0.0,
        posinf=1e12,
        neginf=-1e12,
    )
