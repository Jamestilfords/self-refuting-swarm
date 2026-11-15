import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope encoder representations should lie close to the dominant principal subspace learned from training queries. Residual energy outside that subspace measures semantic variation unsupported by known intents without relying on classifier confidence."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0089"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = x_train.mean(axis=0)
    centered_train = x_train - mean
    centered_eval = x_eval - mean

    cov = centered_train.T @ centered_train
    cov /= max(centered_train.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)

    rank = min(128, vecs.shape[1])
    basis = vecs[:, -rank:]

    train_proj = centered_train @ basis
    eval_proj = centered_eval @ basis

    train_total = np.sum(centered_train * centered_train, axis=1)
    eval_total = np.sum(centered_eval * centered_eval, axis=1)
    train_kept = np.sum(train_proj * train_proj, axis=1)
    eval_kept = np.sum(eval_proj * eval_proj, axis=1)

    train_residual = np.maximum(train_total - train_kept, 0.0)
    eval_residual = np.maximum(eval_total - eval_kept, 0.0)

    scale = np.median(np.abs(train_residual - np.median(train_residual)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.mean(train_residual) + 1e-12

    return np.sqrt(eval_residual + 1e-12) / np.sqrt(scale + 1e-12)
