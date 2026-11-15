import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Queries with substantial encoder-feature mass outside the in-scope principal subspace are likely OOS; combining this ViM-style residual with energy captures both representation and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(
        n_components=64,
        svd_solver="randomized",
        random_state=1729,
        iterated_power=3,
    )
    pca.fit(tr.pooled)

    tr_proj = pca.inverse_transform(pca.transform(tr.pooled))
    ev_proj = pca.inverse_transform(pca.transform(ev.pooled))

    tr_resid = np.linalg.norm(tr.pooled - tr_proj, axis=1)
    ev_resid = np.linalg.norm(ev.pooled - ev_proj, axis=1)

    resid_center = np.median(tr_resid)
    resid_scale = np.median(np.abs(tr_resid - resid_center)) + 1e-6
    residual_score = (ev_resid - resid_center) / resid_scale

    tr_energy = ctx.h.energy(tr.logits)
    energy_center = np.median(tr_energy)
    energy_scale = np.median(np.abs(tr_energy - energy_center)) + 1e-6
    energy_score = (ctx.h.energy(ev.logits) - energy_center) / energy_scale

    return np.nan_to_num(
        energy_score + residual_score,
        nan=0.0,
        posinf=1e12,
        neginf=-1e12,
    )
