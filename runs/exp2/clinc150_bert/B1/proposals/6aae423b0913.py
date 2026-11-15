import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A shrunk diagonal Gaussian models intent-specific feature scales that a tied covariance "
    "misses, while predicted-class calibration makes density deviations comparable across intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0037"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n_classes = tr.logits.shape[1]
    dim = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    sums = np.zeros((n_classes, dim), dtype=np.float64)
    sq_sums = np.zeros((n_classes, dim), dtype=np.float64)
    np.add.at(sums, y, x)
    np.add.at(sq_sums, y, x * x)

    means = np.tile(global_mean, (n_classes, 1))
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    raw_var = np.tile(global_var, (n_classes, 1))
    enough = counts > 1
    raw_var[enough] = (
        sq_sums[enough]
        - sums[enough] * sums[enough] / counts[enough, None]
    ) / (counts[enough, None] - 1.0)

    floor = np.maximum(global_var * 1e-4, 1e-8)
    raw_var = np.maximum(raw_var, floor)

    shrink = (counts / (counts + 24.0))[:, None]
    variances = shrink * raw_var + (1.0 - shrink) * global_var[None, :]
    variances = np.maximum(variances, floor)

    log_var_sum = np.sum(np.log(variances), axis=1)

    def diagonal_nll(a, predicted):
        delta = a - means[predicted]
        mahal = np.sum((delta * delta) / variances[predicted], axis=1)
        return 0.5 * (mahal + log_var_sum[predicted])

    train_pred = np.argmax(tr.logits, axis=1)
    train_nll = diagonal_nll(x, train_pred)

    fallback_mid = np.quantile(train_nll, 0.50)
    fallback_high = np.quantile(train_nll, 0.90)
    mids = np.full(n_classes, fallback_mid, dtype=np.float64)
    highs = np.full(n_classes, fallback_high, dtype=np.float64)

    for c in range(n_classes):
        values = train_nll[train_pred == c]
        if values.size >= 8:
            mids[c] = np.quantile(values, 0.50)
            highs[c] = np.quantile(values, 0.90)

    scales = np.maximum(highs - mids, 1e-6)
    eval_pred = np.argmax(ev.logits, axis=1)
    eval_nll = diagonal_nll(z, eval_pred)
    scores = (eval_nll - mids[eval_pred]) / scales[eval_pred]

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
