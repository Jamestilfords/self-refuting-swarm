import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Known intents have class-specific coordinates whose variation is tightly bounded, so OOS inputs often show a few unusually large standardized residuals even when their overall diagonal Mahalanobis radius is moderate. "
    "A train-calibrated upper-tail score of the largest residual coordinates emphasizes these intent-specific constraint violations."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = x.mean(axis=0)
    global_var = x.var(axis=0) + 1e-8
    counts = np.bincount(y, minlength=k).astype(float)

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)
    shrink_strength = 30.0

    for c in range(k):
        mask = y == c
        n_c = counts[c]
        if n_c <= 0:
            means[c] = global_mean
            variances[c] = global_var
            continue

        xc = x[mask]
        means[c] = xc.mean(axis=0)
        class_var = xc.var(axis=0)
        weight = n_c / (n_c + shrink_strength)
        variances[c] = weight * class_var + (1.0 - weight) * global_var + 1e-8

    top_r = min(8, d)

    def extreme_residual(values, cls):
        residual_sq = (values - means[cls]) ** 2 / variances[cls]
        split = d - top_r
        largest = np.partition(residual_sq, split, axis=1)[:, split:]
        return largest.mean(axis=1)

    train_raw = extreme_residual(x, y)
    centers = np.empty(k, dtype=float)
    scales = np.empty(k, dtype=float)

    global_center = np.median(train_raw)
    global_scale = np.percentile(train_raw, 90.0) - global_center
    global_scale = max(float(global_scale), 1e-6)

    for c in range(k):
        vals = train_raw[y == c]
        if vals.size == 0:
            centers[c] = global_center
            scales[c] = global_scale
        else:
            centers[c] = np.median(vals)
            scales[c] = max(float(np.percentile(vals, 90.0) - centers[c]), 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    eval_raw = extreme_residual(z, pred)
    out = (eval_raw - centers[pred]) / scales[pred]
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
