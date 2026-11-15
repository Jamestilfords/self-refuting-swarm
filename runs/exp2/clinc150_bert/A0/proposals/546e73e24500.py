import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "A ViM-style residual norm measures encoder-feature energy outside the principal in-scope subspace. Queries with semantic variation unsupported by the training representation manifold should have larger residuals."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)

    mean = np.mean(x_train, axis=0)
    centered = x_train - mean

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values * singular_values
    total = np.sum(variance)

    if total > 0.0:
        cumulative = np.cumsum(variance) / total
        rank = int(np.searchsorted(cumulative, 0.90) + 1)
    else:
        rank = 1

    rank = max(1, min(rank, vt.shape[0]))
    basis = vt[:rank]

    eval_centered = x_eval - mean
    projection = eval_centered @ basis.T
    residual_sq = np.sum(eval_centered * eval_centered, axis=1) - np.sum(projection * projection, axis=1)
    return np.sqrt(np.maximum(residual_sq, 0.0))
