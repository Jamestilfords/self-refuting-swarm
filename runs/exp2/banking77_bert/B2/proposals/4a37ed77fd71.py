import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-specific diagonal variances retain intent-dependent feature spread that a tied covariance discards, while classifier-conditioned scoring penalizes queries unsupported by the intent they are assigned to."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0027"]


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

    sq_sums = np.zeros((k, d), dtype=float)
    np.add.at(sq_sums, y, x * x)
    raw_var = sq_sums / np.maximum(counts[:, None], 1.0) - means * means
    raw_var = np.maximum(raw_var, 0.0)

    global_mean = x.mean(axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    shrink = 24.0 / (counts + 24.0)
    variances = (1.0 - shrink[:, None]) * raw_var + shrink[:, None] * global_var
    variances = np.maximum(variances, global_var[None, :] * 1e-4 + 1e-8)

    train_diff = x - means[y]
    train_nll = 0.5 * (
        np.sum(train_diff * train_diff / variances[y], axis=1)
        + np.sum(np.log(variances[y]), axis=1)
    )

    order = np.argsort(np.argmax(tr.logits, axis=1))
    train_pred = np.argmax(tr.logits, axis=1)
    sorted_nll = []
    for c in range(k):
        vals = np.sort(train_nll[train_pred == c])
        if vals.size == 0:
            vals = np.sort(train_nll)
        sorted_nll.append(vals)

    pred = np.argmax(ev.logits, axis=1)
    diff = z - means[pred]
    nll = 0.5 * (
        np.sum(diff * diff / variances[pred], axis=1)
        + np.sum(np.log(variances[pred]), axis=1)
    )

    scores = np.empty(z.shape[0], dtype=float)
    for c in range(k):
        mask = pred == c
        if np.any(mask):
            vals = sorted_nll[c]
            scores[mask] = np.searchsorted(vals, nll[mask], side="right") / float(vals.size)

    return np.clip(scores, 0.0, 1.0)
