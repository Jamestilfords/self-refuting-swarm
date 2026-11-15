import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A shrunk diagonal covariance gives each intent its own stable and unstable feature dimensions "
    "without the noise of a full per-class covariance. Class-calibrated surprise under the classifier's "
    "predicted intent exposes representation-logit disagreement and atypical within-intent variation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0053", "m_0060"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = x.mean(axis=0)
    global_var = x.var(axis=0) + 1e-6
    means = np.empty((k, d), dtype=np.float64)
    variances = np.empty((k, d), dtype=np.float64)
    counts = np.bincount(y, minlength=k).astype(np.float64)

    for c in range(k):
        xc = x[y == c]
        if xc.shape[0] == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = xc.mean(axis=0)
            local_var = xc.var(axis=0)
            shrink = xc.shape[0] / (xc.shape[0] + 32.0)
            variances[c] = shrink * local_var + (1.0 - shrink) * global_var + 1e-6

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    def diagonal_surprise(a, pred):
        out = np.empty(a.shape[0], dtype=np.float64)
        for c in range(k):
            mask = pred == c
            if np.any(mask):
                delta = a[mask] - means[c]
                out[mask] = np.sum(delta * delta / variances[c], axis=1)
        return out

    train_raw = diagonal_surprise(x, tr_pred)
    eval_raw = diagonal_surprise(z, ev_pred)

    med = np.empty(k, dtype=np.float64)
    scale = np.empty(k, dtype=np.float64)
    for c in range(k):
        vals = train_raw[tr_pred == c]
        if vals.size < 8:
            vals = train_raw[y == c]
        if vals.size == 0:
            vals = train_raw
        med[c] = np.median(vals)
        q25, q75 = np.percentile(vals, [25.0, 75.0])
        scale[c] = max(q75 - q25, 1e-6)

    scores = (eval_raw - med[ev_pred]) / scale[ev_pred]
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores + 1e-12 * np.arange(scores.shape[0], dtype=np.float64)
