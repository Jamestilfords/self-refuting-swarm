import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "The fraction of pooled-feature energy outside a train-fitted principal subspace measures "
    "off-manifold variation while discounting overall feature magnitude. Combining its train-calibrated "
    "tail probability with energy highlights queries that are both classifier-uncertain and geometrically atypical."
)
SOURCE = "https://arxiv.org/pdf/2203.10807"
PARENTS = ["energy"]
MEMORY_USED = ["m_0044", "m_0045"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.pooled, dtype=np.float64)
    xev = np.asarray(ev.pooled, dtype=np.float64)

    n_components = min(64, xtr.shape[1], max(1, xtr.shape[0] - 1))
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=314159,
    )
    ztr = pca.fit_transform(xtr)
    zev = pca.transform(xev)

    ctr_sq = np.sum((xtr - pca.mean_) ** 2, axis=1)
    cev_sq = np.sum((xev - pca.mean_) ** 2, axis=1)
    ptr_sq = np.sum(ztr ** 2, axis=1)
    pev_sq = np.sum(zev ** 2, axis=1)

    eps = 1e-10
    rtr = np.maximum(ctr_sq - ptr_sq, 0.0) / np.maximum(ctr_sq, eps)
    rev = np.maximum(cev_sq - pev_sq, 0.0) / np.maximum(cev_sq, eps)

    etr = np.asarray(ctx.h.energy(tr.logits), dtype=np.float64)
    eev = np.asarray(ctx.h.energy(ev.logits), dtype=np.float64)

    r_order = np.sort(np.nan_to_num(rtr, nan=0.0, posinf=1.0, neginf=0.0))
    e_order = np.sort(np.nan_to_num(etr, nan=0.0, posinf=1e6, neginf=-1e6))

    r_tail = np.searchsorted(r_order, np.nan_to_num(rev, nan=0.0), side="right")
    e_tail = np.searchsorted(e_order, np.nan_to_num(eev, nan=0.0), side="right")

    denom = float(max(len(xtr), 1))
    scores = 0.55 * (r_tail / denom) + 0.45 * (e_tail / denom)
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
