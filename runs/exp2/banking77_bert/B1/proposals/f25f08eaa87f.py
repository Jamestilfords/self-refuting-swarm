import numpy as np
from scipy.special import logsumexp

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "ViM-style residual magnitude measures feature energy outside the dominant "
    "in-scope representation subspace, while a train-calibrated virtual OOS logit "
    "compares that residual evidence directly against classifier evidence."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = ["m_0080"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)

    mean = np.mean(x_train, axis=0)
    centered = x_train - mean

    cov = centered.T @ centered / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]

    total = np.sum(values)
    if total > 1e-12:
        cumulative = np.cumsum(values) / total
        rank = int(np.searchsorted(cumulative, 0.90) + 1)
    else:
        rank = 1
    rank = min(max(rank, 1), x_train.shape[1] - 1)

    principal = vectors[:, :rank]

    train_projection = centered @ principal
    train_residual = centered - train_projection @ principal.T
    train_norm = np.linalg.norm(train_residual, axis=1)

    eval_centered = x_eval - mean
    eval_projection = eval_centered @ principal
    eval_residual = eval_centered - eval_projection @ principal.T
    eval_norm = np.linalg.norm(eval_residual, axis=1)

    train_max_logit = np.max(np.asarray(tr.logits, dtype=np.float64), axis=1)
    residual_scale = np.mean(train_norm) + 1e-12
    alpha = np.mean(np.abs(train_max_logit)) / residual_scale
    alpha = max(alpha, 1e-6)

    virtual_logit = alpha * eval_norm
    classifier_log_partition = logsumexp(
        np.asarray(ev.logits, dtype=np.float64), axis=1
    )

    return virtual_logit - classifier_log_partition
