import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "In-scope encoder features should lie near a principal subspace learned from training representations. "
    "OOS queries can retain plausible dominant semantics while exhibiting unusually large residual components."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0125"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    n_comp = max(1, min(128, x_train.shape[0] - 1, x_train.shape[1] - 1))
    pca = PCA(
        n_components=n_comp,
        svd_solver="randomized",
        random_state=271828,
        iterated_power=3,
    )
    pca.fit(x_train)

    centered = x_eval - pca.mean_
    projected = centered @ pca.components_.T
    residual_sq = np.sum(centered * centered, axis=1) - np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(residual_sq, 0.0)

    return np.nan_to_num(
        residual_sq,
        nan=0.0,
        posinf=np.finfo(np.float64).max / 4,
        neginf=0.0,
    )
