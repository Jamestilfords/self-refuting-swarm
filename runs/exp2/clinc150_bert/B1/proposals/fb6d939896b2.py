import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "In-scope representations should lie near the dominant principal subspace of training features, "
    "whereas OOS inputs can retain larger residual components in the complementary space."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_components = min(256, tr.pooled.shape[0], tr.pooled.shape[1])

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=271828,
        iterated_power=3,
    )
    pca.fit(tr.pooled)

    projected = pca.transform(ev.pooled)
    reconstruction = projected @ pca.components_ + pca.mean_
    residual = ev.pooled - reconstruction
    scores = np.sqrt(np.sum(residual * residual, axis=1))

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
