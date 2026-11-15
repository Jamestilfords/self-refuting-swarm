import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query assigned to an intent should also be plausible under that intent's own diagonal feature distribution, including dimensions whose variance is unusually tight for that class. Shrinkage and predicted-class robust calibration stabilize class-specific surprise for sparse intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0045", "m_0057"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + 1e-6
    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)

    for c in range(k):
        xc = x[y == c]
        n_c = xc.shape[0]
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            if n_c > 1:
                class_var = np.var(xc, axis=0)
            else:
                class_var = global_var
            shrink = n_c / (n_c + 20.0)
            variances[c] = shrink * class_var + (1.0 - shrink) * global_var + 1e-6

    def conditional_surprise(features, logits):
        pred = np.argmax(logits, axis=1)
        mu = means[pred]
        var = variances[pred]
        residual = features - mu
        return 0.5 * np.sum((residual * residual) / var + np.log(var), axis=1)

    train_raw = conditional_surprise(x, tr.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_raw = conditional_surprise(z, ev.logits)
    eval_pred = np.argmax(ev.logits, axis=1)

    centers = np.empty(k, dtype=float)
    scales = np.empty(k, dtype=float)
    fallback_center = np.median(train_raw)
    fallback_scale = np.quantile(train_raw, 0.75) - np.quantile(train_raw, 0.25)
    fallback_scale = max(float(fallback_scale), 1e-6)

    for c in range(k):
        values = train_raw[train_pred == c]
        if values.size < 8:
            centers[c] = fallback_center
            scales[c] = fallback_scale
        else:
            centers[c] = np.median(values)
            iqr = np.quantile(values, 0.75) - np.quantile(values, 0.25)
            scales[c] = max(float(iqr), 1e-6)

    return (eval_raw - centers[eval_pred]) / scales[eval_pred]
