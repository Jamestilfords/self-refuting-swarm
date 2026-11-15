import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class eigen-directions encode tight representation constraints that known intents satisfy. "
    "The maximum whitened deviation in these trailing directions detects a localized constraint violation that residual-energy sums can dilute."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0105", "m_0113"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    global_mean = x.mean(axis=0)
    means = np.tile(global_mean, (k, 1))
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    centered = x - means[y]
    cov = centered.T @ centered / max(float(x.shape[0] - k), 1.0)
    cov = 0.5 * (cov + cov.T)

    vals, vecs = np.linalg.eigh(cov)
    r = min(48, d)
    tail_vals = np.maximum(vals[:r], 0.0)
    positive = tail_vals[tail_vals > 1e-12]
    if positive.size:
        floor = max(float(np.percentile(positive, 20.0)) * 0.05, 1e-8)
    else:
        floor = 1e-8
    scale = np.sqrt(np.maximum(tail_vals, floor))
    basis = vecs[:, :r]

    train_pred = np.argmax(tr.logits, axis=1)
    train_proj = (x - means[train_pred]) @ basis
    train_raw = np.max(np.abs(train_proj) / scale[None, :], axis=1)

    ev_pred = np.argmax(ev.logits, axis=1)
    ev_proj = (np.asarray(ev.pooled, dtype=np.float64) - means[ev_pred]) @ basis
    ev_raw = np.max(np.abs(ev_proj) / scale[None, :], axis=1)

    global_sorted = np.sort(train_raw)
    scores = np.empty(ev_raw.shape[0], dtype=np.float64)
    for c in range(k):
        mask = ev_pred == c
        if not np.any(mask):
            continue
        ref = train_raw[train_pred == c]
        if ref.size < 8:
            ref = global_sorted
        else:
            ref = np.sort(ref)
        scores[mask] = np.searchsorted(ref, ev_raw[mask], side="right") / float(ref.size)

    return np.nan_to_num(scores + 1e-8 * np.tanh(ev_raw), nan=0.0, posinf=1.0, neginf=0.0)
