import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Residual energy outside the dominant encoder subspace captures feature variation unsupported "
    "by known intents, while an energy-dependent gate emphasizes that residual for uncertain queries."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0131", "m_0143"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    n_components = min(96, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=314159,
    )
    pca.fit(tr.pooled)

    tr_centered = tr.pooled - pca.mean_
    ev_centered = ev.pooled - pca.mean_
    tr_codes = pca.transform(tr.pooled)
    ev_codes = pca.transform(ev.pooled)

    tr_resid_sq = np.maximum(
        np.sum(tr_centered * tr_centered, axis=1) - np.sum(tr_codes * tr_codes, axis=1),
        0.0,
    )
    ev_resid_sq = np.maximum(
        np.sum(ev_centered * ev_centered, axis=1) - np.sum(ev_codes * ev_codes, axis=1),
        0.0,
    )

    tr_resid = np.sqrt(tr_resid_sq)
    ev_resid = np.sqrt(ev_resid_sq)

    resid_med = float(np.median(tr_resid))
    resid_scale = float(np.median(np.abs(tr_resid - resid_med))) + 1e-8
    resid_z = (ev_resid - resid_med) / resid_scale

    tr_energy = ctx.h.energy(tr.logits)
    ev_energy = ctx.h.energy(ev.logits)
    energy_med = float(np.median(tr_energy))
    energy_scale = float(np.median(np.abs(tr_energy - energy_med))) + 1e-8
    energy_z = (ev_energy - energy_med) / energy_scale

    gate = 1.0 / (1.0 + np.exp(-np.clip(energy_z, -30.0, 30.0)))
    scores = resid_z * (0.5 + gate) + 0.35 * energy_z

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
