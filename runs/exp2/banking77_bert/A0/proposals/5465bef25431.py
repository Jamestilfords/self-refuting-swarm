import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Classifier-relevant feature directions are learned from train-only feature/logit cross-covariance. "
    "OOS queries should retain unusually large standardized feature energy outside this discriminative subspace."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0023"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mean = np.mean(tr.pooled, axis=0)
    scale = np.std(tr.pooled, axis=0)
    scale = np.maximum(scale, 1e-6)

    x_train = (tr.pooled - mean) / scale
    x_eval = (ev.pooled - mean) / scale

    centered_logits = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    cross_cov = x_train.T @ centered_logits
    u, _, _ = np.linalg.svd(cross_cov, full_matrices=False)

    rank = min(32, u.shape[1])
    directions = u[:, :rank]

    train_proj_sq = np.sum((x_train @ directions) ** 2, axis=1)
    eval_proj_sq = np.sum((x_eval @ directions) ** 2, axis=1)

    train_residual = np.maximum(np.sum(x_train * x_train, axis=1) - train_proj_sq, 0.0)
    eval_residual = np.maximum(np.sum(x_eval * x_eval, axis=1) - eval_proj_sq, 0.0)

    center = np.median(train_residual)
    spread = np.quantile(train_residual, 0.9) - np.quantile(train_residual, 0.1)
    return (eval_residual - center) / max(spread, 1e-6)
