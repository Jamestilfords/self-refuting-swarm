import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Low-variance within-intent directions capture deviations that normal in-scope variation rarely expresses. Calibrating trailing-subspace residual energy relative to the predicted intent's training distribution highlights class-specific off-manifold queries."
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

    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    centered = x - means[y]
    cov = (centered.T @ centered) / max(len(x) - k, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n_tail = min(96, max(16, d // 8))
    tail_basis = eigvecs[:, :n_tail]
    tail_scale = np.maximum(eigvals[:n_tail], 1e-8)

    train_proj = centered @ tail_basis
    train_tail = np.sum((train_proj * train_proj) / tail_scale, axis=1)

    class_loc = np.zeros(k, dtype=float)
    class_spread = np.ones(k, dtype=float)
    for c in range(k):
        vals = train_tail[y == c]
        if len(vals):
            class_loc[c] = np.median(vals)
            class_spread[c] = np.median(np.abs(vals - class_loc[c])) + 1e-6

    pred = np.argmax(ev.logits, axis=1)
    ev_centered = z - means[pred]
    ev_proj = ev_centered @ tail_basis
    ev_tail = np.sum((ev_proj * ev_proj) / tail_scale, axis=1)

    scores = (ev_tail - class_loc[pred]) / class_spread[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
