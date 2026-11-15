import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A predicted intent should also be plausible under that intent's own feature-wise variability. "
    "Per-class diagonal shrinkage exposes atypical deviations that tied covariance can wash out."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    xe = np.asarray(ev.pooled, dtype=float)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + 1e-6

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)

    for c in range(k):
        xc = x[y == c]
        nc = len(xc)
        if nc == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            class_var = np.var(xc, axis=0)
            shrink = min(1.0, 20.0 / max(nc, 1))
            variances[c] = (1.0 - shrink) * class_var + shrink * global_var + 1e-6

    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)

    def diagonal_surprise(z, pred):
        out = np.empty(len(z), dtype=float)
        for c in range(k):
            mask = pred == c
            if np.any(mask):
                delta = z[mask] - means[c]
                out[mask] = np.sum(
                    (delta * delta) / variances[c] + np.log(variances[c]),
                    axis=1,
                )
        return out

    train_surprise = diagonal_surprise(x, pred_train)
    eval_surprise = diagonal_surprise(xe, pred_eval)

    out = np.empty(len(xe), dtype=float)
    global_sorted = np.sort(train_surprise)
    for c in range(k):
        mask = pred_eval == c
        if not np.any(mask):
            continue
        reference = np.sort(train_surprise[pred_train == c])
        if len(reference) < 8:
            reference = global_sorted
        ranks = np.searchsorted(reference, eval_surprise[mask], side="right")
        out[mask] = ranks / (len(reference) + 1.0)

    return out
