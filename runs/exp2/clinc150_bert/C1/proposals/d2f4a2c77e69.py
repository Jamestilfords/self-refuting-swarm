import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Out-of-scope inputs tend to have substantial encoder-feature mass outside the principal subspace learned from in-scope training data. Combining this residual norm with energy captures both representation-level novelty and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_comp = min(128, tr.pooled.shape[0], tr.pooled.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=17)
    z_train = pca.fit_transform(tr.pooled)
    z_eval = pca.transform(ev.pooled)

    train_centered = tr.pooled - pca.mean_
    eval_centered = ev.pooled - pca.mean_

    train_total = np.sum(train_centered * train_centered, axis=1)
    eval_total = np.sum(eval_centered * eval_centered, axis=1)
    train_projected = np.sum(z_train * z_train, axis=1)
    eval_projected = np.sum(z_eval * z_eval, axis=1)

    train_residual = np.sqrt(np.maximum(train_total - train_projected, 0.0))
    eval_residual = np.sqrt(np.maximum(eval_total - eval_projected, 0.0))

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    r_med = np.median(train_residual)
    e_med = np.median(train_energy)
    r_scale = np.median(np.abs(train_residual - r_med)) * 1.4826
    e_scale = np.median(np.abs(train_energy - e_med)) * 1.4826

    r_scale = max(float(r_scale), 1e-6)
    e_scale = max(float(e_scale), 1e-6)

    scores = (eval_residual - r_med) / r_scale
    scores += 0.25 * (eval_energy - e_med) / e_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
