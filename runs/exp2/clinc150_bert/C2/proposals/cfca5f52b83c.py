import numpy as np
from sklearn.utils.extmath import randomized_svd

FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope encoder representations tend to lie near the dominant training-feature subspace, while OOS queries often retain a larger residual component outside that subspace."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mean = np.mean(tr.pooled, axis=0, keepdims=True)
    centered_train = tr.pooled - mean

    _, _, components = randomized_svd(
        centered_train,
        n_components=128,
        n_oversamples=16,
        n_iter=3,
        random_state=1729,
    )

    centered_eval = ev.pooled - mean
    projection = centered_eval @ components.T
    total_sq = np.sum(centered_eval * centered_eval, axis=1)
    projected_sq = np.sum(projection * projection, axis=1)
    residual_sq = np.maximum(total_sq - projected_sq, 0.0)

    return np.sqrt(residual_sq)
