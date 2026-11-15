import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "In-scope encoder representations tend to lie near a principal subspace learned from training data, while OOS inputs retain larger orthogonal residuals. Adding a train-calibrated residual score to energy combines representation-space novelty with classifier uncertainty."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_comp = min(64, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=2718)
    z_train = pca.fit_transform(tr.pooled)
    z_eval = pca.transform(ev.pooled)

    train_recon = pca.inverse_transform(z_train)
    eval_recon = pca.inverse_transform(z_eval)
    train_residual = np.linalg.norm(tr.pooled - train_recon, axis=1)
    eval_residual = np.linalg.norm(ev.pooled - eval_recon, axis=1)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    residual_center = np.median(train_residual)
    energy_center = np.median(train_energy)
    residual_scale = np.median(np.abs(train_residual - residual_center))
    energy_scale = np.median(np.abs(train_energy - energy_center))

    if residual_scale < 1e-8:
        residual_scale = np.std(train_residual) + 1e-8
    if energy_scale < 1e-8:
        energy_scale = np.std(train_energy) + 1e-8

    residual_score = (eval_residual - residual_center) / residual_scale
    energy_score = (eval_energy - energy_center) / energy_scale
    return residual_score + energy_score
