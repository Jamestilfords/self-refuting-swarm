import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query should be close in the feature directions that are stable for the intent predicted by the classifier. This uses shrunk per-class diagonal variances and emphasizes the largest standardized residual directions rather than ordinary Gaussian density."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    d = x_train.shape[1]
    k = tr.logits.shape[1]
    global_mean = np.mean(x_train, axis=0)
    global_var = np.mean((x_train - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-6)

    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))

    for c in range(k):
        xc = x_train[labels == c]
        n_c = xc.shape[0]
        if n_c > 0:
            mean_c = np.mean(xc, axis=0)
            if n_c > 1:
                var_c = np.mean((xc - mean_c) ** 2, axis=0)
            else:
                var_c = global_var
            shrink = 32.0 / (n_c + 32.0)
            means[c] = mean_c
            variances[c] = np.maximum(
                (1.0 - shrink) * var_c + shrink * global_var,
                1e-6,
            )

    predicted = np.argmax(ev.logits, axis=1)
    scores = np.empty(x_eval.shape[0], dtype=float)
    tail_count = max(1, d // 8)

    for c in np.unique(predicted):
        idx = np.flatnonzero(predicted == c)
        diff = x_eval[idx] - means[c]
        residual = (diff * diff) / variances[c]
        largest = np.partition(residual, d - tail_count, axis=1)[:, -tail_count:]
        tail_residual = np.mean(largest, axis=1)
        scale_penalty = np.mean(np.log(variances[c] / global_var))
        scores[idx] = tail_residual + 0.25 * scale_penalty

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
