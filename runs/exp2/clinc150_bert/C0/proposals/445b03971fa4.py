import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Queries outside the training feature manifold retain substantial norm after projection away from the principal in-scope subspace. Combining this ViM-style residual with energy captures both representation-space novelty and classifier uncertainty."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_comp = min(128, tr.pooled.shape[1], tr.pooled.shape[0] - 1)
    pca = PCA(n_components=max(1, n_comp), svd_solver="randomized", random_state=17)
    pca.fit(tr.pooled)

    def residual_norm(x):
        centered = x - pca.mean_
        total_sq = np.sum(centered * centered, axis=1)
        projected = centered @ pca.components_.T
        kept_sq = np.sum(projected * projected, axis=1)
        return np.sqrt(np.maximum(total_sq - kept_sq, 0.0))

    train_resid = residual_norm(tr.pooled)
    eval_resid = residual_norm(ev.pooled)

    resid_med = np.median(train_resid)
    resid_scale = np.median(np.abs(train_resid - resid_med)) * 1.4826
    resid_scale = max(float(resid_scale), 1e-6)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    energy_med = np.median(train_energy)
    energy_scale = np.median(np.abs(train_energy - energy_med)) * 1.4826
    energy_scale = max(float(energy_scale), 1e-6)

    residual_score = (eval_resid - resid_med) / resid_scale
    energy_score = (np.asarray(ctx.h.energy(ev.logits), dtype=float) - energy_med) / energy_scale
    return np.nan_to_num(residual_score + 0.35 * energy_score, nan=0.0, posinf=1e6, neginf=-1e6)
