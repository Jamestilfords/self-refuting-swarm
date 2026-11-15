import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions are constrained for supported intents, and an OOS shift can "
    "produce a coherent excursion across several such directions. Random signed sketches detect these "
    "coherent trailing-subspace shifts rather than reducing them to a Mahalanobis radius."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0006", "m_0017", "m_0087"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    n_classes = tr.logits.shape[1]
    d = x.shape[1]
    trail_dim = min(64, max(16, d // 4))
    sketch_dim = 24
    rng = np.random.default_rng(271828)

    means = np.zeros((n_classes, d), dtype=np.float64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    for c in range(n_classes):
        mask = y == c
        if np.any(mask):
            means[c] = x[mask].mean(axis=0)
    global_mean = x.mean(axis=0)
    means[counts == 0] = global_mean

    residuals = x - means[y]
    cov = residuals.T @ residuals / max(len(x) - n_classes, 1)
    cov.flat[:: d + 1] += 1e-6 * (np.trace(cov) / max(d, 1) + 1.0)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[:trail_dim]
    basis = vecs[:, order]
    scales = np.sqrt(np.maximum(vals[order], 1e-10))

    signs = rng.choice(np.array([-1.0, 1.0]), size=(trail_dim, sketch_dim))
    signs /= np.sqrt(float(trail_dim))

    train_tail = ((x - means[y]) @ basis) / scales
    train_sketch = train_tail @ signs

    loc = np.zeros((n_classes, sketch_dim), dtype=np.float64)
    spread = np.ones((n_classes, sketch_dim), dtype=np.float64)
    for c in range(n_classes):
        mask = pred_tr == c
        if np.sum(mask) >= 4:
            vals_c = train_sketch[mask]
        else:
            vals_c = train_sketch
        loc[c] = np.median(vals_c, axis=0)
        mad = np.median(np.abs(vals_c - loc[c]), axis=0)
        spread[c] = np.maximum(1.4826 * mad, 0.15)

    scores = np.empty(len(z), dtype=np.float64)
    for c in range(n_classes):
        mask = pred_ev == c
        if not np.any(mask):
            continue
        tail = ((z[mask] - means[c]) @ basis) / scales
        sketch = tail @ signs
        standardized = np.abs((sketch - loc[c]) / spread[c])
        scores[mask] = np.mean(np.log1p(np.maximum(standardized - 1.0, 0.0)), axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
