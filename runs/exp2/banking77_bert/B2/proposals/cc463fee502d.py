import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A global principal subspace captures dominant in-scope feature variation, while OOS queries often retain unusually large residual feature energy outside that subspace. Train-calibrated residual energy is combined with train-calibrated energy so neither scale dominates."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], tr.pooled.shape[0] - 1)
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=917)
    pca.fit(tr.pooled)

    def residual_norm(x):
        centered = x - pca.mean_
        total_sq = np.sum(centered * centered, axis=1)
        projected = pca.transform(x)
        explained_sq = np.sum(projected * projected, axis=1)
        return np.sqrt(np.maximum(total_sq - explained_sq, 0.0))

    train_residual = residual_norm(tr.pooled)
    eval_residual = residual_norm(ev.pooled)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    res_med = np.median(train_residual)
    eng_med = np.median(train_energy)
    res_scale = np.median(np.abs(train_residual - res_med)) + 1e-6
    eng_scale = np.median(np.abs(train_energy - eng_med)) + 1e-6

    residual_score = (eval_residual - res_med) / res_scale
    energy_score = (eval_energy - eng_med) / eng_scale
    return residual_score + energy_score
