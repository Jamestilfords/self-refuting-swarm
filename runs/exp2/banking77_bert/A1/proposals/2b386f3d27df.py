import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent should also have competitive likelihood under a "
    "class-specific diagonal covariance model with shrinkage for sparse classes. "
    "OOS queries can be confidently assigned to one intent while fitting another intent's feature geometry better."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0057", "m_0069"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0)
    global_var = np.maximum(global_var, 1e-6)

    means = np.empty((n_classes, dim), dtype=float)
    variances = np.empty((n_classes, dim), dtype=float)

    for c in range(n_classes):
        members = x_train[labels == c]
        count = members.shape[0]
        if count == 0:
            means[c] = global_mean
            variances[c] = global_var
            continue

        mean = np.mean(members, axis=0)
        if count > 1:
            class_var = np.var(members, axis=0)
        else:
            class_var = global_var

        shrink = 32.0 / (count + 32.0)
        means[c] = mean
        variances[c] = np.maximum(
            (1.0 - shrink) * class_var + shrink * global_var,
            1e-6,
        )

    predicted = np.argmax(ev.logits, axis=1)
    best_nll = np.full(x_eval.shape[0], np.inf, dtype=float)
    predicted_nll = np.empty(x_eval.shape[0], dtype=float)

    for c in range(n_classes):
        diff = x_eval - means[c]
        nll = 0.5 * (
            np.sum((diff * diff) / variances[c], axis=1)
            + np.sum(np.log(variances[c]))
        )
        best_nll = np.minimum(best_nll, nll)
        mask = predicted == c
        if np.any(mask):
            predicted_nll[mask] = nll[mask]

    score_out = predicted_nll - best_nll
    score_out = score_out + 1e-3 * predicted_nll
    return np.nan_to_num(score_out, nan=0.0, posinf=1e12, neginf=-1e12)
