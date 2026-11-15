import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Training features concentrate near a principal in-scope subspace, so a large orthogonal residual identifies representations that are poorly explained by in-scope variation. Combining this ViM-style residual with energy captures both feature-space novelty and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_comp = max(1, min(128, tr.pooled.shape[0] - 1, tr.pooled.shape[1]))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=314159)
    pca.fit(tr.pooled)

    def residual_norm(x):
        z = pca.transform(x)
        recon = pca.inverse_transform(z)
        return np.sqrt(np.sum((x - recon) ** 2, axis=1))

    train_resid = residual_norm(tr.pooled)
    eval_resid = residual_norm(ev.pooled)
    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    resid_med = np.median(train_resid)
    resid_scale = np.median(np.abs(train_resid - resid_med)) * 1.4826 + 1e-8
    energy_med = np.median(train_energy)
    energy_scale = np.median(np.abs(train_energy - energy_med)) * 1.4826 + 1e-8

    scores = ((eval_resid - resid_med) / resid_scale +
              (eval_energy - energy_med) / energy_scale)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
