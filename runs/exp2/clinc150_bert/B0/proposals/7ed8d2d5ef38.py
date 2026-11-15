import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Queries outside the training feature manifold retain substantial energy in directions "
    "discarded by a principal subspace, while energy contributes complementary classifier uncertainty."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    tr_centered = tr.pooled - pca.mean_
    ev_centered = ev.pooled - pca.mean_
    tr_proj = pca.transform(tr.pooled)
    ev_proj = pca.transform(ev.pooled)

    tr_residual = np.sqrt(np.maximum(
        np.sum(tr_centered * tr_centered, axis=1) - np.sum(tr_proj * tr_proj, axis=1),
        0.0,
    ))
    ev_residual = np.sqrt(np.maximum(
        np.sum(ev_centered * ev_centered, axis=1) - np.sum(ev_proj * ev_proj, axis=1),
        0.0,
    ))

    r_med = np.median(tr_residual)
    r_scale = np.percentile(tr_residual, 75) - np.percentile(tr_residual, 25)
    r_scale = max(float(r_scale), 1e-6)

    tr_energy = ctx.h.energy(tr.logits)
    ev_energy = ctx.h.energy(ev.logits)
    e_med = np.median(tr_energy)
    e_scale = np.percentile(tr_energy, 75) - np.percentile(tr_energy, 25)
    e_scale = max(float(e_scale), 1e-6)

    return (ev_residual - r_med) / r_scale + 0.25 * (ev_energy - e_med) / e_scale
