import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Within-class low-variance directions are tightly constrained for known intents, "
    "so unusually large trailing-subspace deviation relative to leading variation is OOS-like."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0010"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)

    n, d = x.shape
    k = tr.logits.shape[1]
    global_mean = x.mean(axis=0)

    means = np.tile(global_mean, (k, 1))
    valid = (y >= 0) & (y < k)
    for c in range(k):
        xc = x[valid & (y == c)]
        if xc.shape[0] > 0:
            means[c] = xc.mean(axis=0)

    centered = x - means[np.clip(y, 0, k - 1)]
    cov = (centered.T @ centered) / max(n - k, 1)
    base = float(np.trace(cov) / max(d, 1))
    cov.flat[::d + 1] += max(base * 1e-6, 1e-8)

    vals, vecs = np.linalg.eigh(cov)
    floor = max(float(vals[-1]) * 1e-8, 1e-10)
    vals = np.maximum(vals, floor)

    q = max(8, d // 4)
    tail_vals = vals[:q]
    tail_vecs = vecs[:, :q]
    head_vals = vals[-q:]
    head_vecs = vecs[:, -q:]

    pred = np.argmax(ev.logits, axis=1)
    tail = np.empty(z.shape[0], dtype=np.float64)
    head = np.empty(z.shape[0], dtype=np.float64)

    for c in range(k):
        idx = np.flatnonzero(pred == c)
        if idx.size == 0:
            continue
        delta = z[idx] - means[c]
        tail_proj = delta @ tail_vecs
        head_proj = delta @ head_vecs
        tail[idx] = np.sum((tail_proj * tail_proj) / tail_vals, axis=1)
        head[idx] = np.sum((head_proj * head_proj) / head_vals, axis=1)

    scores = np.log1p(tail / q) - np.log1p(head / q)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
