import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent should explain a query under that intent's own "
    "semantic tolerances, with per-class variances shrunk toward global variances for stability."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0051", "m_0062"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)
    pred_eval = np.argmax(ev.logits, axis=1).astype(np.int64)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    global_mean = x_train.mean(axis=0)
    global_var = x_train.var(axis=0) + 1e-6

    means = np.empty((n_classes, dim), dtype=np.float64)
    variances = np.empty((n_classes, dim), dtype=np.float64)
    train_nll = np.empty(x_train.shape[0], dtype=np.float64)

    for c in range(n_classes):
        idx = labels == c
        count = int(idx.sum())

        if count == 0:
            means[c] = global_mean
            variances[c] = global_var
            continue

        xc = x_train[idx]
        mean_c = xc.mean(axis=0)
        var_c = xc.var(axis=0)

        # Empirical-Bayes diagonal shrinkage prevents tiny intents from producing
        # unstable precision estimates while retaining broad/narrow intent geometry.
        weight = count / (count + 32.0)
        var_c = weight * var_c + (1.0 - weight) * global_var
        var_c = np.maximum(var_c, 1e-6)

        means[c] = mean_c
        variances[c] = var_c

        residual = xc - mean_c
        train_nll[idx] = 0.5 * np.sum(
            residual * residual / var_c + np.log(var_c), axis=1
        )

    eval_nll = np.empty(x_eval.shape[0], dtype=np.float64)
    for c in range(n_classes):
        idx = pred_eval == c
        if not np.any(idx):
            continue
        residual = x_eval[idx] - means[c]
        eval_nll[idx] = 0.5 * np.sum(
            residual * residual / variances[c] + np.log(variances[c]), axis=1
        )

    # Class-conditional empirical tail probability makes scores comparable across
    # intents with intrinsically different diagonal covariance volumes.
    out = np.empty(x_eval.shape[0], dtype=np.float64)
    for c in range(n_classes):
        idx = pred_eval == c
        if not np.any(idx):
            continue
        reference = np.sort(train_nll[labels == c])
        if reference.size == 0:
            reference = np.sort(train_nll)
        ranks = np.searchsorted(reference, eval_nll[idx], side="right")
        out[idx] = ranks / (reference.size + 1.0)

    return np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
