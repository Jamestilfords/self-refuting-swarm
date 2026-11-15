import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A scale-normalized residual ratio measures how much encoder-feature energy lies outside the dominant in-scope representation subspace. Combining its train-calibrated tail score with energy captures both geometric novelty and classifier uncertainty."
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0135", "m_0147"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    rank = min(128, tr.pooled.shape[1], max(2, tr.pooled.shape[0] - 1))
    pca = PCA(n_components=rank, svd_solver="randomized", random_state=2718)
    pca.fit(tr.pooled)

    def residual_ratio(x):
        centered = x - pca.mean_
        projected = pca.inverse_transform(pca.transform(x)) - pca.mean_
        residual = centered - projected
        num = np.sum(residual * residual, axis=1)
        den = np.sum(centered * centered, axis=1) + 1e-12
        return num / den

    train_resid = residual_ratio(tr.pooled)
    eval_resid = residual_ratio(ev.pooled)

    resid_center = np.median(train_resid)
    resid_scale = np.median(np.abs(train_resid - resid_center)) + 1e-12
    energy_train = ctx.h.energy(tr.logits)
    energy_center = np.median(energy_train)
    energy_scale = np.median(np.abs(energy_train - energy_center)) + 1e-12

    resid_tail = 1.0 / (1.0 + np.exp(-np.clip(
        (eval_resid - resid_center) / (1.4826 * resid_scale), -30.0, 30.0
    )))
    energy_tail = 1.0 / (1.0 + np.exp(-np.clip(
        (ctx.h.energy(ev.logits) - energy_center) / (1.4826 * energy_scale), -30.0, 30.0
    )))

    return 0.65 * resid_tail + 0.35 * energy_tail
