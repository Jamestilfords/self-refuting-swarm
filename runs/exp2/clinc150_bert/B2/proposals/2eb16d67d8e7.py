import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Out-of-subspace residual energy measures how much a pooled representation violates the dominant global in-scope feature geometry. OOS queries should retain larger components in low-variance directions even when a classifier assigns them a confident intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0078"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = x_train.mean(axis=0)
    centered_train = x_train - mean
    centered_eval = x_eval - mean

    cov = (centered_train.T @ centered_train) / max(len(centered_train) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    total = np.maximum(vals.sum(), 1e-12)
    cumulative = np.cumsum(np.maximum(vals, 0.0)) / total
    dim = int(np.searchsorted(cumulative, 0.90) + 1)
    dim = max(8, min(dim, 256, vecs.shape[1]))

    basis = vecs[:, :dim]
    projection = centered_eval @ basis
    residual_sq = np.sum(centered_eval * centered_eval, axis=1) - np.sum(projection * projection, axis=1)
    residual_sq = np.maximum(residual_sq, 0.0)

    train_projection = centered_train @ basis
    train_residual_sq = np.sum(centered_train * centered_train, axis=1) - np.sum(train_projection * train_projection, axis=1)
    train_residual_sq = np.maximum(train_residual_sq, 0.0)

    median = np.median(train_residual_sq)
    scale = np.median(np.abs(train_residual_sq - median)) + 1e-8
    return (residual_sq - median) / scale
