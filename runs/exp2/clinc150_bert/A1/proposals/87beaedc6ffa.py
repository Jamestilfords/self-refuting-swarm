import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "OOS queries can contain encoder variation outside the dominant in-scope representation subspace. Combining PCA residual magnitude with energy preserves classifier uncertainty while adding a feature-space novelty signal."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0049"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_comp = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    center = pca.mean_
    basis = pca.components_

    tr_centered = tr.pooled - center
    ev_centered = ev.pooled - center

    tr_proj = tr_centered @ basis.T
    ev_proj = ev_centered @ basis.T

    tr_resid = np.sum(tr_centered * tr_centered, axis=1) - np.sum(tr_proj * tr_proj, axis=1)
    ev_resid = np.sum(ev_centered * ev_centered, axis=1) - np.sum(ev_proj * ev_proj, axis=1)
    tr_resid = np.maximum(tr_resid, 0.0)
    ev_resid = np.maximum(ev_resid, 0.0)

    tr_energy = ctx.h.energy(tr.logits)
    ev_energy = ctx.h.energy(ev.logits)

    resid_scale = max(float(np.std(tr_resid)), 1e-8)
    energy_scale = max(float(np.std(tr_energy)), 1e-8)

    resid_z = (ev_resid - float(np.mean(tr_resid))) / resid_scale
    energy_z = (ev_energy - float(np.mean(tr_energy))) / energy_scale

    return np.nan_to_num(energy_z + 0.35 * resid_z, nan=0.0, posinf=1e12, neginf=-1e12)
