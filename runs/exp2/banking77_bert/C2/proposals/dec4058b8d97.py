import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Out-of-scope queries often contain substantial encoder-feature energy outside the principal subspace learned from in-scope training examples. The residual norm therefore measures deviation from the dominant in-scope representation manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mean = np.mean(tr.pooled, axis=0)
    centered_train = tr.pooled - mean

    cov = (centered_train.T @ centered_train) / max(len(centered_train) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    rank = min(128, eigvecs.shape[1])
    basis = eigvecs[:, -rank:]

    centered_eval = ev.pooled - mean
    total_sq = np.sum(centered_eval * centered_eval, axis=1)
    projected = centered_eval @ basis
    explained_sq = np.sum(projected * projected, axis=1)

    residual_sq = np.maximum(total_sq - explained_sq, 0.0)
    return np.sqrt(residual_sq + 1e-12)
