import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class residual directions are tightly constrained, but OOS shifts may be coherent along oblique combinations rather than a single eigen-coordinate. "
    "A fixed random-projection scan detects these unsupported trailing-subspace deviations and calibrates them separately for the classifier-selected intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0188", "m_0263"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    global_mean = x.mean(axis=0)
    means = np.empty((k, d), dtype=np.float64)
    present = counts > 0
    means[present] = sums[present] / counts[present, None]
    means[~present] = global_mean

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x), 1)
    scale = float(np.trace(cov) / max(d, 1))
    cov.flat[::d + 1] += max(scale * 1e-5, 1e-8)

    vals, vecs = np.linalg.eigh(cov)
    r = min(96, d)
    vals = np.maximum(vals[:r], 1e-8)
    basis = vecs[:, :r] / np.sqrt(vals)[None, :]

    rng = np.random.default_rng(20250308)
    directions = rng.normal(size=(r, 64))
    directions /= np.maximum(
        np.sqrt(np.sum(directions * directions, axis=0, keepdims=True)),
        1e-12,
    )

    train_pred = np.argmax(tr.logits, axis=1)
    train_white = (x - means[train_pred]) @ basis
    train_scan = np.max(np.abs(train_white @ directions), axis=1)

    eval_pred = np.argmax(ev.logits, axis=1)
    eval_white = (z - means[eval_pred]) @ basis
    eval_scan = np.max(np.abs(eval_white @ directions), axis=1)

    sorted_scans = []
    fallback = np.sort(train_scan)
    for c in range(k):
        values = np.sort(train_scan[train_pred == c])
        sorted_scans.append(values if len(values) >= 8 else fallback)

    out = np.empty(len(z), dtype=np.float64)
    for c in range(k):
        mask = eval_pred == c
        if np.any(mask):
            ref = sorted_scans[c]
            rank = np.searchsorted(ref, eval_scan[mask], side="left")
            survival = (len(ref) - rank + 1.0) / (len(ref) + 1.0)
            out[mask] = -np.log(np.maximum(survival, 1e-12))

    return np.nan_to_num(out, nan=0.0, posinf=30.0, neginf=0.0)
