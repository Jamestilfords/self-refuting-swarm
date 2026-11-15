import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-assigned intent should have not only a plausible mean representation but also "
    "plausible coordinate-wise variation; class-specific shrunk diagonal variances preserve intent-dependent feature scales."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    floor = 1e-6 * (float(np.mean(global_var)) + 1.0)
    global_var = np.maximum(global_var, floor)

    sums = np.zeros((k, d), dtype=float)
    sums_sq = np.zeros((k, d), dtype=float)
    np.add.at(sums, y, x)
    np.add.at(sums_sq, y, x * x)

    denom = np.maximum(counts[:, None], 1.0)
    raw_mean = sums / denom
    raw_var = np.maximum(sums_sq / denom - raw_mean * raw_mean, 0.0)

    mean_weight = counts / (counts + 8.0)
    var_weight = counts / (counts + 32.0)
    means = mean_weight[:, None] * raw_mean + (1.0 - mean_weight[:, None]) * global_mean
    variances = (
        var_weight[:, None] * raw_var
        + (1.0 - var_weight[:, None]) * global_var
    )
    variances = np.maximum(variances, floor)
    log_vars = np.log(variances)

    train_diff = x - means[y]
    train_nll = 0.5 * np.sum(
        train_diff * train_diff / variances[y] + log_vars[y],
        axis=1,
    )

    cal_mean_sum = np.bincount(y, weights=train_nll, minlength=k)
    cal_mean = cal_mean_sum / np.maximum(counts, 1.0)
    centered = train_nll - cal_mean[y]
    cal_var = np.bincount(y, weights=centered * centered, minlength=k)
    cal_var /= np.maximum(counts, 1.0)
    cal_std = np.sqrt(np.maximum(cal_var, 1e-6))

    pred = np.argmax(ev.logits, axis=1)
    diff = z - means[pred]
    eval_nll = 0.5 * np.sum(
        diff * diff / variances[pred] + log_vars[pred],
        axis=1,
    )
    return (eval_nll - cal_mean[pred]) / cal_std[pred]
