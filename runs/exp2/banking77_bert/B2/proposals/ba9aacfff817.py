import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Residual energy outside the principal in-scope encoder subspace identifies semantic content "
    "not represented by known intents, while energy adds classifier-side uncertainty evidence."
)
SOURCE = "https://openaccess.thecvf.com/content/CVPR2022/html/Wang_ViM_Out-of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.html"
PARENTS = ["energy"]
MEMORY_USED = ["m_0031", "m_0045"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_components = min(128, tr.pooled.shape[1], tr.pooled.shape[0] - 1)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    tr_proj = pca.transform(tr.pooled)
    ev_proj = pca.transform(ev.pooled)
    tr_recon = tr_proj @ pca.components_ + pca.mean_
    ev_recon = ev_proj @ pca.components_ + pca.mean_

    tr_residual = np.sqrt(np.maximum(np.sum((tr.pooled - tr_recon) ** 2, axis=1), 0.0))
    ev_residual = np.sqrt(np.maximum(np.sum((ev.pooled - ev_recon) ** 2, axis=1), 0.0))

    tr_energy = ctx.h.energy(tr.logits)
    ev_energy = ctx.h.energy(ev.logits)

    r_lo, r_hi = np.quantile(tr_residual, [0.1, 0.9])
    e_lo, e_hi = np.quantile(tr_energy, [0.1, 0.9])
    r_scale = max(float(r_hi - r_lo), 1e-8)
    e_scale = max(float(e_hi - e_lo), 1e-8)

    residual_score = (ev_residual - r_lo) / r_scale
    energy_score = (ev_energy - e_lo) / e_scale
    return residual_score + 0.35 * energy_score
