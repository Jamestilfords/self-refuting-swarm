import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions are tightly constrained, but OOS inputs can "
    "produce an atypical cross-direction residual-shape distribution rather than only a large radius."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0188", "m_0220"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n, d = x.shape
    k = tr.logits.shape[1]
    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    residual = x - means[y]
    cov = residual.T @ residual / max(n - k, 1)
    cov.flat[::d + 1] += 1e-6 * (np.trace(cov) / max(d, 1) + 1.0)

    vals, vecs = np.linalg.eigh(cov)
    q = min(64, max(8, d // 4))
    idx = np.argsort(vals)[:q]
    basis = vecs[:, idx]
    scale = np.sqrt(np.maximum(vals[idx], 1e-8))

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    def shape_stat(a, pred):
        r = (a - means[pred]) @ basis
        r = r / scale
        r = np.clip(r, -20.0, 20.0)
        m2 = np.mean(r * r, axis=1)
        m4 = np.mean(r ** 4, axis=1)
        kurt = m4 / np.maximum(m2 * m2, 1e-10)
        return np.abs(kurt - 3.0)

    train_stat = shape_stat(x, train_pred)
    eval_stat = shape_stat(z, eval_pred)
    global_sorted = np.sort(train_stat)
    out = np.empty(len(z), dtype=float)

    for c in range(k):
        mask = train_pred == c
        target = eval_pred == c
        if not np.any(target):
            continue
        ref = np.sort(train_stat[mask]) if np.sum(mask) >= 12 else global_sorted
        pos = np.searchsorted(ref, eval_stat[target], side="right")
        lower = pos / (len(ref) + 1.0)
        upper = 1.0 - lower
        out[target] = -np.log(np.maximum(np.minimum(lower, upper), 1e-12))

    return np.nan_to_num(out, nan=0.0, posinf=30.0, neginf=0.0)
