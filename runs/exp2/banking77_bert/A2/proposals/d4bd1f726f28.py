import numpy as np
from sklearn.utils.extmath import randomized_svd

FAMILY = "other"
ARM = "ce"
RATIONALE = "A ViM-style residual norm measures encoder energy outside the principal in-scope feature subspace, while the maximum logit discounts residuals from confidently supported intents. OOS queries should have both larger residual components and weaker classifier support."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    center = np.mean(x_train, axis=0)

    centered_train = x_train - center
    rank = min(128, centered_train.shape[0] - 1, centered_train.shape[1])
    if rank < 1:
        return -np.max(ev.logits, axis=1)

    _, _, components = randomized_svd(
        centered_train,
        n_components=rank,
        n_iter=4,
        random_state=2718,
    )

    train_proj = centered_train @ components.T
    train_residual = np.sqrt(
        np.maximum(
            np.sum(centered_train * centered_train, axis=1)
            - np.sum(train_proj * train_proj, axis=1),
            0.0,
        )
    )

    centered_eval = x_eval - center
    eval_proj = centered_eval @ components.T
    eval_residual = np.sqrt(
        np.maximum(
            np.sum(centered_eval * centered_eval, axis=1)
            - np.sum(eval_proj * eval_proj, axis=1),
            0.0,
        )
    )

    train_max_logit = np.max(tr.logits, axis=1)
    denom = np.mean(train_residual)
    alpha = np.mean(np.abs(train_max_logit)) / max(denom, 1e-12)

    return alpha * eval_residual - np.max(ev.logits, axis=1)
