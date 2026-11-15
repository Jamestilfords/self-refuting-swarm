import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A ViM-style residual norm measures feature energy outside the principal in-scope representation subspace. Combining this residual with energy captures semantic components unexplained by the training manifold while retaining classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))

    pca = PCA(
        n_components=dim,
        svd_solver="randomized",
        random_state=271828,
    )
    pca.fit(tr.pooled)

    def residual_norm(x):
        centered = x - pca.mean_
        projected = pca.transform(x)
        total_sq = np.sum(centered * centered, axis=1)
        explained_sq = np.sum(projected * projected, axis=1)
        return np.sqrt(np.maximum(total_sq - explained_sq, 0.0))

    train_residual = residual_norm(tr.pooled)
    eval_residual = residual_norm(ev.pooled)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    residual_scale = float(np.std(train_residual))
    energy_scale = float(np.std(train_energy))
    if not np.isfinite(residual_scale) or residual_scale < 1e-8:
        residual_scale = 1.0
    if not np.isfinite(energy_scale) or energy_scale < 1e-8:
        energy_scale = 1.0

    scores = eval_residual / residual_scale + eval_energy / energy_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
