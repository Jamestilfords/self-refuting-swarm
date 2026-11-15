import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ViM-style residual magnitude measures how far a bottleneck feature lies outside the principal in-scope feature subspace. Adding a train-calibrated energy term combines representation novelty with classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_comp = min(128, tr.feats.shape[1], max(1, tr.feats.shape[0] - 1))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=17)
    pca.fit(tr.feats)

    tr_proj = pca.inverse_transform(pca.transform(tr.feats))
    ev_proj = pca.inverse_transform(pca.transform(ev.feats))

    tr_residual = np.linalg.norm(tr.feats - tr_proj, axis=1)
    ev_residual = np.linalg.norm(ev.feats - ev_proj, axis=1)

    tr_energy = ctx.h.energy(tr.logits)
    res_scale = np.std(tr_residual) + 1e-8
    energy_scale = np.std(tr_energy) + 1e-8

    scores = ev_residual / res_scale + 0.1 * ctx.h.energy(ev.logits) / energy_scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
