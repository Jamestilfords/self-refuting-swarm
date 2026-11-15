import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration vector captures the shape of evidential support across intents, "
    "which can be atypical for OOS queries even when total evidence or vacuity appears in-distribution."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0098", "m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def clr(alpha):
        z = np.log(np.maximum(alpha, 1e-8))
        return z - z.mean(axis=1, keepdims=True)

    train_shape = clr(ctx.h.alpha(tr.logits))
    eval_shape = clr(ctx.h.alpha(ev.logits))

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    k = tr.logits.shape[1]
    global_mean = train_shape.mean(axis=0)
    global_var = train_shape.var(axis=0) + 1e-6

    means = np.empty((k, train_shape.shape[1]))
    variances = np.empty_like(means)

    for c in range(k):
        xc = train_shape[train_pred == c]
        n = xc.shape[0]
        if n == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            weight = n / (n + 12.0)
            class_mean = xc.mean(axis=0)
            class_var = xc.var(axis=0) + 1e-6
            means[c] = weight * class_mean + (1.0 - weight) * global_mean
            variances[c] = weight * class_var + (1.0 - weight) * global_var

    residual = eval_shape - means[eval_pred]
    score_values = np.mean(
        (residual * residual) / variances[eval_pred],
        axis=1,
    )
    return np.nan_to_num(score_values, nan=0.0, posinf=1e12, neginf=0.0)
