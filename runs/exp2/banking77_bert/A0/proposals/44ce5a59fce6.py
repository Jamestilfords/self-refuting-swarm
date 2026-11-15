import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "In-scope bottleneck features concentrate near a dominant training-feature subspace, whereas OOS inputs often retain larger components in its orthogonal residual space. A train-calibrated residual norm provides a ViM-style OOS signal without relying on evaluation-set statistics."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)

    mean = x_train.mean(axis=0, keepdims=True)
    centered = x_train - mean

    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    total = float(eigvals.sum())
    if total > 1e-12:
        rank = int(np.searchsorted(np.cumsum(eigvals) / total, 0.90) + 1)
    else:
        rank = 1
    rank = min(max(rank, 1), x_train.shape[1] - 1)

    basis = eigvecs[:, :rank]

    train_norm_sq = np.sum(centered * centered, axis=1)
    train_proj_sq = np.sum((centered @ basis) ** 2, axis=1)
    train_residual = np.sqrt(np.maximum(train_norm_sq - train_proj_sq, 0.0))

    eval_centered = x_eval - mean
    eval_norm_sq = np.sum(eval_centered * eval_centered, axis=1)
    eval_proj_sq = np.sum((eval_centered @ basis) ** 2, axis=1)
    eval_residual = np.sqrt(np.maximum(eval_norm_sq - eval_proj_sq, 0.0))

    median = np.median(train_residual)
    mad = np.median(np.abs(train_residual - median))
    scale = max(1.4826 * mad, np.std(train_residual), 1e-8)

    return (eval_residual - median) / scale
