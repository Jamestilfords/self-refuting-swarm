import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Rather than using only the residual norm, this ViM-style score detects a single unusually large standardized coordinate in the residual feature subspace, capturing sparse unsupported semantic deviations that can be hidden by aggregate residual energy."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0147", "m_0042"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = np.mean(x_train, axis=0)
    centered = x_train - mean
    n, d = centered.shape

    cov = (centered.T @ centered) / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]

    rank = min(max(1, d // 3), d - 1)
    residual_idx = order[rank:]
    residual_basis = eigvecs[:, residual_idx]
    residual_var = np.maximum(eigvals[residual_idx], 1e-10)

    train_residual = centered @ residual_basis
    train_standardized = train_residual / np.sqrt(residual_var)
    train_stat = np.max(np.abs(train_standardized), axis=1)

    median = np.median(train_stat)
    q25, q75 = np.percentile(train_stat, [25.0, 75.0])
    scale = max(q75 - q25, 1e-8)

    eval_residual = (x_eval - mean) @ residual_basis
    eval_standardized = eval_residual / np.sqrt(residual_var)
    eval_stat = np.max(np.abs(eval_standardized), axis=1)

    scores = (eval_stat - median) / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
