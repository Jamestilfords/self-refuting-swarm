import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions are tightly constrained, but their residual signs can "
    "also be intent-specific. A predicted-intent sign pseudo-likelihood flags OOS examples whose "
    "low-variance residual pattern is unsupported even when their total trailing energy is ordinary."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0017", "m_0188", "m_0006"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    global_mean = x.mean(axis=0)
    means = np.zeros((k, d), dtype=np.float64)
    for c in range(k):
        if counts[c] > 0:
            means[c] = x[y == c].mean(axis=0)
        else:
            means[c] = global_mean

    centered = x - means[y]
    cov = centered.T @ centered / max(len(x) - k, 1)
    ridge = 1e-5 * (np.trace(cov) / max(d, 1) + 1e-8)
    vals, vecs = np.linalg.eigh(cov + ridge * np.eye(d))

    q = min(96, d)
    vals = np.maximum(vals[:q], ridge)
    basis = vecs[:, :q]
    scale = np.sqrt(vals)

    train_coords = (centered @ basis) / scale
    alpha = 0.75
    pos = np.full((k, q), alpha, dtype=np.float64)
    neg = np.full((k, q), alpha, dtype=np.float64)
    for c in range(k):
        if counts[c] > 0:
            signs = train_coords[y == c] > 0.0
            pos[c] += signs.sum(axis=0)
            neg[c] += (~signs).sum(axis=0)
    p_pos = pos / (pos + neg)

    train_signs = train_coords > 0.0
    train_nll = -np.mean(
        np.where(train_signs, np.log(p_pos[y]), np.log1p(-p_pos[y])),
        axis=1,
    )

    med = np.zeros(k, dtype=np.float64)
    mad = np.ones(k, dtype=np.float64)
    for c in range(k):
        vals_c = train_nll[y == c]
        if len(vals_c):
            med[c] = np.median(vals_c)
            mad[c] = np.median(np.abs(vals_c - med[c])) + 1e-6

    pred = np.argmax(ev.logits, axis=1).astype(np.int64)
    eval_coords = ((z - means[pred]) @ basis) / scale
    eval_signs = eval_coords > 0.0
    nll = -np.mean(
        np.where(eval_signs, np.log(p_pos[pred]), np.log1p(-p_pos[pred])),
        axis=1,
    )
    return (nll - med[pred]) / mad[pred]
