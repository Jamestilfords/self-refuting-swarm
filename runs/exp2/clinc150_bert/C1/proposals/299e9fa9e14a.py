import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual norm measures encoder components unsupported by the principal in-scope feature subspace, while standardized energy captures classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], tr.pooled.shape[0] - 1)
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=17)
    z_train = pca.fit_transform(tr.pooled)
    z_eval = pca.transform(ev.pooled)

    train_recon = pca.inverse_transform(z_train)
    eval_recon = pca.inverse_transform(z_eval)
    r_train = np.sqrt(np.sum((tr.pooled - train_recon) ** 2, axis=1))
    r_eval = np.sqrt(np.sum((ev.pooled - eval_recon) ** 2, axis=1))

    e_train = ctx.h.energy(tr.logits)
    e_eval = ctx.h.energy(ev.logits)

    r_scale = np.std(r_train) + 1e-8
    e_scale = np.std(e_train) + 1e-8
    return (r_eval - np.mean(r_train)) / r_scale + (e_eval - np.mean(e_train)) / e_scale
