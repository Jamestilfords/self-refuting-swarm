import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A global principal subspace captures common in-scope encoder variation, while the residual norm measures feature content unsupported by that subspace. Combining train-calibrated residual novelty with energy retains classifier uncertainty information for samples that remain close to the broad in-scope manifold."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0073"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    pca = PCA(
        n_components=128,
        svd_solver="randomized",
        random_state=2718,
        whiten=False,
    )
    pca.fit(tr.pooled)

    tr_proj = pca.inverse_transform(pca.transform(tr.pooled))
    ev_proj = pca.inverse_transform(pca.transform(ev.pooled))

    tr_residual = np.sum((tr.pooled - tr_proj) ** 2, axis=1)
    ev_residual = np.sum((ev.pooled - ev_proj) ** 2, axis=1)

    tr_energy = ctx.h.energy(tr.logits)
    ev_energy = ctx.h.energy(ev.logits)

    r_med = np.median(tr_residual)
    r_scale = np.quantile(tr_residual, 0.75) - np.quantile(tr_residual, 0.25)
    e_med = np.median(tr_energy)
    e_scale = np.quantile(tr_energy, 0.75) - np.quantile(tr_energy, 0.25)

    r_scale = max(float(r_scale), 1e-8)
    e_scale = max(float(e_scale), 1e-8)

    residual_score = 1.0 / (1.0 + np.exp(-np.clip((ev_residual - r_med) / r_scale, -30, 30)))
    energy_score = 1.0 / (1.0 + np.exp(-np.clip((ev_energy - e_med) / e_scale, -30, 30)))

    return 0.65 * residual_score + 0.35 * energy_score
