import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions encode tightly constrained in-scope variation. "
    "A maximum standardized residual over these low-variance directions detects sparse constraint violations that a summed distance can dilute."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0139"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    z = np.asarray(ev.pooled, dtype=float)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    global_mean = x.mean(axis=0)
    means = np.zeros((k, d), dtype=float)
    for c in range(k):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)
        else:
            means[c] = global_mean

    centered = x - means[y]
    cov = (centered.T @ centered) / max(len(x) - k, 1)
    scale = float(np.trace(cov) / max(d, 1))
    cov.flat[:: d + 1] += max(scale, 1e-8) * 1e-5

    vals, vecs = np.linalg.eigh(cov)
    n_tail = min(64, d)
    vals = np.maximum(vals[:n_tail], max(scale, 1e-8) * 1e-6)
    basis = vecs[:, :n_tail]

    train_proj = (centered @ basis) / np.sqrt(vals)
    train_stat = np.max(np.abs(train_proj), axis=1)

    pred = np.argmax(ev.logits, axis=1)
    eval_centered = z - means[pred]
    eval_proj = (eval_centered @ basis) / np.sqrt(vals)
    eval_stat = np.max(np.abs(eval_proj), axis=1)

    out = np.empty(len(z), dtype=float)
    for c in range(k):
        mask = pred == c
        if not np.any(mask):
            continue
        reference = np.sort(train_stat[y == c])
        if reference.size < 2:
            reference = np.sort(train_stat)
        ranks = np.searchsorted(reference, eval_stat[mask], side="right")
        out[mask] = ranks / (reference.size + 1.0)

    return out
