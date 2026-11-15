import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A class-specific diagonal covariance captures intent dimensions with different semantic tolerance, "
    "while shrinkage toward the global variance stabilizes rare intents. Predicted-intent tail calibration "
    "flags examples whose variance-normalized residual is unusually large for the class they are routed to."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0103", "m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    pred_eval = np.asarray(ev.logits).argmax(axis=1)

    n, dim = x_train.shape
    classes = int(max(np.max(labels), np.max(pred_eval))) + 1

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-6

    means = np.empty((classes, dim), dtype=float)
    variances = np.empty((classes, dim), dtype=float)
    medians = np.empty(classes, dtype=float)
    scales = np.empty(classes, dtype=float)

    for c in range(classes):
        mask = labels == c
        xc = x_train[mask]
        count = xc.shape[0]

        if count == 0:
            means[c] = global_mean
            variances[c] = global_var
            train_dist = np.sum(
                (x_train - global_mean) ** 2 / global_var, axis=1
            ) / dim
        else:
            mean_c = np.mean(xc, axis=0)
            var_c = np.var(xc, axis=0)
            shrink = count / (count + 32.0)

            means[c] = mean_c
            variances[c] = shrink * var_c + (1.0 - shrink) * global_var + 1e-6
            train_dist = np.sum(
                (xc - means[c]) ** 2 / variances[c], axis=1
            ) / dim

        medians[c] = np.median(train_dist)
        upper = np.quantile(train_dist, 0.90)
        scales[c] = max(upper - medians[c], 1e-6)

    chosen_mean = means[pred_eval]
    chosen_var = variances[pred_eval]
    residual = np.sum((x_eval - chosen_mean) ** 2 / chosen_var, axis=1) / dim

    tail = np.maximum(
        (residual - medians[pred_eval]) / scales[pred_eval], 0.0
    )
    return np.log1p(np.nan_to_num(tail, nan=0.0, posinf=1e6, neginf=0.0))
