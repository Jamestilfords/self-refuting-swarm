import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Out-of-subspace residual norm measures encoder components unsupported by the in-scope principal feature subspace, while energy adds classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_comp = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=17)
    pca.fit(tr.pooled)

    tr_proj = pca.inverse_transform(pca.transform(tr.pooled))
    ev_proj = pca.inverse_transform(pca.transform(ev.pooled))

    tr_resid = np.linalg.norm(tr.pooled - tr_proj, axis=1)
    ev_resid = np.linalg.norm(ev.pooled - ev_proj, axis=1)

    tr_energy = ctx.h.energy(tr.logits)
    ev_energy = ctx.h.energy(ev.logits)

    resid_center = np.median(tr_resid)
    resid_scale = np.median(np.abs(tr_resid - resid_center)) + 1e-8
    energy_center = np.median(tr_energy)
    energy_scale = np.median(np.abs(tr_energy - energy_center)) + 1e-8

    scores = ((ev_resid - resid_center) / resid_scale +
              (ev_energy - energy_center) / energy_scale)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
