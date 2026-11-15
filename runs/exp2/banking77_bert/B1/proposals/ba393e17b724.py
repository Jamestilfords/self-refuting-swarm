import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A predicted intent should place an in-scope query inside that intent's own feature spread, "
    "while OOS queries often violate class-specific low-variance constraints. Shrinking each "
    "diagonal class covariance toward the pooled variance stabilizes this test for sparse intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    sums = np.zeros((k, d), dtype=float)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    global_mean = x.mean(axis=0)
    pooled_var = ((x - global_mean) ** 2).mean(axis=0)
    pooled_var = np.maximum(pooled_var, 1e-8)

    sq_sums = np.zeros((k, d), dtype=float)
    np.add.at(sq_sums, y, x * x)
    raw_var = sq_sums / np.maximum(counts[:, None], 1.0) - means * means
    raw_var = np.maximum(raw_var, 0.0)

    # Small classes borrow more of the stable pooled diagonal covariance.
    shrink = 20.0 / (counts + 20.0)
    variances = (1.0 - shrink[:, None]) * raw_var + shrink[:, None] * pooled_var
    variances = np.maximum(variances, pooled_var[None, :] * 1e-5 + 1e-8)

    train_diff = x - means[y]
    train_dist = np.sum((train_diff * train_diff) / variances[y], axis=1)

    center = np.zeros(k, dtype=float)
    scale = np.zeros(k, dtype=float)
    default_scale = max(float(np.sqrt(d)), 1.0)
    for c in range(k):
        dc = train_dist[y == c]
        if dc.size:
            center[c] = np.quantile(dc, 0.90)
            spread = np.quantile(dc, 0.75) - np.quantile(dc, 0.25)
            scale[c] = max(float(spread), default_scale * 0.1)
        else:
            center[c] = float(d)
            scale[c] = default_scale

    predicted = np.argmax(ev.logits, axis=1)
    eval_diff = z - means[predicted]
    eval_dist = np.sum(
        (eval_diff * eval_diff) / variances[predicted],
        axis=1,
    )
    scores = (eval_dist - center[predicted]) / scale[predicted]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
