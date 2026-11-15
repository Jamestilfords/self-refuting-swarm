import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A ViM-style residual norm measures representation energy outside the dominant in-scope pooled-feature subspace, where OOS queries should be less well explained. Train-calibrated energy adds complementary classifier uncertainty without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    dim = min(128, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=137)
    pca.fit(tr.pooled)

    tr_proj = pca.transform(tr.pooled)
    ev_proj = pca.transform(ev.pooled)

    tr_centered_norm2 = np.sum((tr.pooled - pca.mean_) ** 2, axis=1)
    ev_centered_norm2 = np.sum((ev.pooled - pca.mean_) ** 2, axis=1)
    tr_residual = np.sqrt(np.maximum(tr_centered_norm2 - np.sum(tr_proj ** 2, axis=1), 0.0))
    ev_residual = np.sqrt(np.maximum(ev_centered_norm2 - np.sum(ev_proj ** 2, axis=1), 0.0))

    tr_energy = ctx.h.energy(tr.logits)
    ev_energy = ctx.h.energy(ev.logits)

    res_med = np.median(tr_residual)
    res_scale = np.median(np.abs(tr_residual - res_med)) * 1.4826
    res_scale = max(float(res_scale), 1e-6)

    en_med = np.median(tr_energy)
    en_scale = np.median(np.abs(tr_energy - en_med)) * 1.4826
    en_scale = max(float(en_scale), 1e-6)

    residual_z = np.clip((ev_residual - res_med) / res_scale, -20.0, 20.0)
    energy_z = np.clip((ev_energy - en_med) / en_scale, -20.0, 20.0)

    return residual_z + 0.35 * energy_z
