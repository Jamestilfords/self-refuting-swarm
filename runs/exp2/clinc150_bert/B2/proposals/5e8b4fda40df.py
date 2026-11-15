import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Features far from the training principal subspace have a large residual norm, while energy captures classifier uncertainty. Train-calibrated robust scaling lets these complementary signals contribute without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(n_components=128, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    train_proj = pca.transform(tr.pooled)
    eval_proj = pca.transform(ev.pooled)

    train_recon = train_proj @ pca.components_ + pca.mean_
    eval_recon = eval_proj @ pca.components_ + pca.mean_

    train_resid = np.sqrt(np.sum((tr.pooled - train_recon) ** 2, axis=1))
    eval_resid = np.sqrt(np.sum((ev.pooled - eval_recon) ** 2, axis=1))

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    resid_med = np.median(train_resid)
    energy_med = np.median(train_energy)
    resid_scale = np.quantile(train_resid, 0.75) - np.quantile(train_resid, 0.25)
    energy_scale = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)

    resid_scale = max(float(resid_scale), 1e-8)
    energy_scale = max(float(energy_scale), 1e-8)

    scores = (eval_resid - resid_med) / resid_scale
    scores += (eval_energy - energy_med) / energy_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
