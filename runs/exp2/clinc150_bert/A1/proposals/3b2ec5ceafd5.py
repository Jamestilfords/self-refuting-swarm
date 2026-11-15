import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A predicted intent should assign a representation a typical class-conditional diagonal "
    "likelihood after accounting for that intent's own coordinate variances. Shrinkage and "
    "per-class train-calibrated surprise avoid favoring broad intents merely because they have larger spread."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0087"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)

    k = tr.logits.shape[1]
    d = x.shape[1]
    eps = 1e-8

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, eps)

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)
    counts = np.zeros(k, dtype=int)

    for c in range(k):
        xc = x[y == c]
        n_c = xc.shape[0]
        counts[c] = n_c
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var
            continue

        mean_c = np.mean(xc, axis=0)
        var_c = np.mean((xc - mean_c) ** 2, axis=0)
        # Small classes borrow coordinate-wise variance from the in-scope population.
        weight = n_c / (n_c + 32.0)
        means[c] = mean_c
        variances[c] = np.maximum(weight * var_c + (1.0 - weight) * global_var, eps)

    train_nll = np.empty(x.shape[0], dtype=float)
    for c in range(k):
        idx = y == c
        if not np.any(idx):
            continue
        diff = x[idx] - means[c]
        train_nll[idx] = 0.5 * np.sum(
            diff * diff / variances[c] + np.log(variances[c]), axis=1
        )

    nll_mean = np.zeros(k, dtype=float)
    nll_scale = np.ones(k, dtype=float)
    overall_mean = float(np.mean(train_nll))
    overall_scale = float(np.std(train_nll) + eps)
    for c in range(k):
        vals = train_nll[y == c]
        if vals.size >= 2:
            nll_mean[c] = np.mean(vals)
            nll_scale[c] = max(float(np.std(vals)), eps)
        else:
            nll_mean[c] = overall_mean
            nll_scale[c] = overall_scale

    pred = np.argmax(ev.logits, axis=1)
    diff = z - means[pred]
    var = variances[pred]
    eval_nll = 0.5 * np.sum(diff * diff / var + np.log(var), axis=1)

    standardized = (eval_nll - nll_mean[pred]) / nll_scale[pred]
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=50.0, neginf=-50.0)
    return np.clip(standardized, -50.0, 50.0)
