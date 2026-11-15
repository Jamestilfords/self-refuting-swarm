import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent should respect that intent's coordinate-wise feature "
    "stability, while sparse large standardized residuals are amplified rather than averaged away."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n, d = x_train.shape
    k = tr.logits.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-6

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)
    train_raw = [None] * k

    def raw_tail(x, mean, var):
        z2 = (x - mean) ** 2 / var
        return np.mean(np.logaddexp(0.0, z2 - 4.0), axis=1)

    for c in range(k):
        xc = x_train[labels == c]
        nc = len(xc)
        if nc == 0:
            means[c] = global_mean
            variances[c] = global_var
            train_raw[c] = np.array([0.0], dtype=float)
            continue

        mean_c = np.mean(xc, axis=0)
        var_c = np.mean((xc - mean_c) ** 2, axis=0)
        shrink = nc / (nc + 20.0)

        means[c] = mean_c
        variances[c] = shrink * var_c + (1.0 - shrink) * global_var + 1e-6
        train_raw[c] = np.sort(raw_tail(xc, means[c], variances[c]))

    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(len(x_eval), dtype=float)

    for c in range(k):
        mask = pred == c
        if not np.any(mask):
            continue
        raw = raw_tail(x_eval[mask], means[c], variances[c])
        reference = train_raw[c]
        tail_rank = np.searchsorted(reference, raw, side="right") / len(reference)
        out[mask] = tail_rank + 1e-6 * np.tanh(raw)

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)
