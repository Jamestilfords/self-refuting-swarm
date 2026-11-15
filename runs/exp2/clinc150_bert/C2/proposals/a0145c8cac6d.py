import numpy as np
from sklearn.utils.extmath import randomized_svd

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope representations should lie close to the dominant principal subspace of training encoder features, while OOS queries often have a larger residual component. This is a ViM-style residual-space novelty score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    mean = np.mean(x_train, axis=0, keepdims=True)
    centered_train = x_train - mean

    max_rank = min(centered_train.shape[0], centered_train.shape[1])
    n_components = min(128, max(1, max_rank - 1))

    _, _, vt = randomized_svd(
        centered_train,
        n_components=n_components,
        n_iter=4,
        random_state=2718,
    )

    centered_eval = x_eval - mean
    projection = centered_eval @ vt.T
    total_sq = np.sum(centered_eval * centered_eval, axis=1)
    projected_sq = np.sum(projection * projection, axis=1)
    residual_sq = np.maximum(total_sq - projected_sq, 0.0)

    return np.sqrt(residual_sq + 1e-12)
