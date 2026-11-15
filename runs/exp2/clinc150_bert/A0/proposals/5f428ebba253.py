import numpy as np
from scipy.special import logsumexp

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A ViM-style virtual OOS logit converts the norm outside the dominant training-feature "
    "subspace into competing evidence against known-class logits. OOS inputs should have larger "
    "unsupported residual components and therefore receive more virtual-class probability."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0049"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)

    mean = x_train.mean(axis=0)
    train_centered = x_train - mean
    eval_centered = x_eval - mean

    cov = (train_centered.T @ train_centered) / max(len(x_train) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]

    total_var = vals.sum()
    if total_var > 1e-12:
        cumulative = np.cumsum(vals) / total_var
        rank = int(np.searchsorted(cumulative, 0.90) + 1)
    else:
        rank = 1
    rank = min(max(rank, 1), vecs.shape[1] - 1)

    basis = vecs[:, :rank]

    train_residual = train_centered - (train_centered @ basis) @ basis.T
    eval_residual = eval_centered - (eval_centered @ basis) @ basis.T
    train_norm = np.sqrt(np.maximum(np.sum(train_residual * train_residual, axis=1), 0.0))
    eval_norm = np.sqrt(np.maximum(np.sum(eval_residual * eval_residual, axis=1), 0.0))

    known_peak = np.max(np.asarray(tr.logits, dtype=float), axis=1)
    mean_residual = float(np.mean(train_norm))
    positive_peak = float(np.mean(np.maximum(known_peak, 0.0)))
    alpha = positive_peak / max(mean_residual, 1e-8)
    alpha = max(alpha, 1e-4)

    virtual_logit = alpha * eval_norm
    known_log_partition = logsumexp(np.asarray(ev.logits, dtype=float), axis=1)

    scores = virtual_logit - known_log_partition
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
