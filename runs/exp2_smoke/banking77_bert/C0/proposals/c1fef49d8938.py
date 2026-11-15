import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Queries with substantial encoder-feature energy outside the principal in-scope subspace should be OOS. Combining this ViM-style residual with energy captures both representation novelty and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_comp = min(128, tr.pooled.shape[0] - 1, tr.pooled.shape[1])

    if n_comp >= 1:
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=2718)
        pca.fit(tr.pooled)

        tr_z = pca.transform(tr.pooled)
        ev_z = pca.transform(ev.pooled)
        tr_centered_sq = np.sum((tr.pooled - pca.mean_) ** 2, axis=1)
        ev_centered_sq = np.sum((ev.pooled - pca.mean_) ** 2, axis=1)
        tr_residual = np.sqrt(np.maximum(tr_centered_sq - np.sum(tr_z ** 2, axis=1), 0.0))
        ev_residual = np.sqrt(np.maximum(ev_centered_sq - np.sum(ev_z ** 2, axis=1), 0.0))
    else:
        center = np.mean(tr.pooled, axis=0, keepdims=True)
        tr_residual = np.linalg.norm(tr.pooled - center, axis=1)
        ev_residual = np.linalg.norm(ev.pooled - center, axis=1)

    tr_energy = ctx.h.energy(tr.logits)
    ev_energy = ctx.h.energy(ev.logits)

    res_med = np.median(tr_residual)
    eng_med = np.median(tr_energy)
    res_scale = np.median(np.abs(tr_residual - res_med)) * 1.4826
    eng_scale = np.median(np.abs(tr_energy - eng_med)) * 1.4826

    if not np.isfinite(res_scale) or res_scale < 1e-8:
        res_scale = 1.0
    if not np.isfinite(eng_scale) or eng_scale < 1e-8:
        eng_scale = 1.0

    scores = (ev_residual - res_med) / res_scale + (ev_energy - eng_med) / eng_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
