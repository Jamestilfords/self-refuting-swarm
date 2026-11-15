import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode constraints that in-scope examples rarely violate. "
    "A predicted-class residual measured only in these trailing directions exposes samples that are incompatible with the assigned intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0007", "m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = np.mean(x, axis=0)

    means = np.empty((k, d), dtype=float)
    for c in range(k):
        members = x[y == c]
        means[c] = np.mean(members, axis=0) if len(members) else global_mean

    centered = x - means[y]
    cov = (centered.T @ centered) / max(n - k, 1)
    scale = float(np.trace(cov) / max(d, 1))
    cov.flat[::d + 1] += max(scale, 1e-8) * 1e-5

    eigvals, eigvecs = np.linalg.eigh(cov)
    tail_dim = max(8, min(d // 4, 64))
    tail_vals = np.maximum(eigvals[:tail_dim], max(scale, 1e-8) * 1e-5)
    tail_vecs = eigvecs[:, :tail_dim]

    train_pred = np.argmax(tr.logits, axis=1)
    train_resid = (x - means[train_pred]) @ tail_vecs
    train_raw = np.sum((train_resid * train_resid) / tail_vals, axis=1)

    medians = np.zeros(k, dtype=float)
    scales = np.ones(k, dtype=float)
    for c in range(k):
        values = train_raw[train_pred == c]
        if len(values):
            medians[c] = np.median(values)
            mad = np.median(np.abs(values - medians[c]))
            scales[c] = max(1.4826 * mad, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    resid = (z - means[pred]) @ tail_vecs
    raw = np.sum((resid * resid) / tail_vals, axis=1)
    out = (raw - medians[pred]) / scales[pred]

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
