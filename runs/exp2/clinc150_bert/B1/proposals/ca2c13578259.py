import numpy as np
from scipy.linalg import eigh

FAMILY = "density"
ARM = "ce"
RATIONALE = "Within-class low-variance directions encode constraints that ordinary intent examples satisfy, but OOS examples may violate through one or a few unusually large residual coordinates. A maximum standardized trailing-subspace deviation detects localized constraint violations rather than summing the same residual energy used by prior proposals."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0126"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    n, d = x_train.shape
    k = tr.logits.shape[1]
    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(labels, minlength=k).astype(float)

    for c in range(k):
        mask = labels == c
        if np.any(mask):
            means[c] = x_train[mask].mean(axis=0)

    global_mean = x_train.mean(axis=0)
    means[counts == 0] = global_mean

    scatter = np.zeros((d, d), dtype=float)
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            residual = x_train[mask] - means[c]
            scatter += residual.T @ residual

    denom = max(n - np.count_nonzero(counts), 1)
    cov = scatter / float(denom)
    ridge = max(float(np.trace(cov)) / max(d, 1), 1e-8) * 1e-5
    cov.flat[::d + 1] += ridge

    tail_dim = min(64, d)
    vals, vecs = eigh(
        cov,
        subset_by_index=[0, tail_dim - 1],
        check_finite=False
    )
    vals = np.maximum(vals, ridge)
    inv_scale = 1.0 / np.sqrt(vals)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    def localized_tail(x, pred):
        residual = x - means[pred]
        coords = (residual @ vecs) * inv_scale
        squared = coords * coords
        return np.max(squared, axis=1) + 0.05 * np.mean(squared, axis=1)

    train_stat = localized_tail(x_train, train_pred)
    eval_stat = localized_tail(x_eval, eval_pred)

    global_ref = np.sort(np.nan_to_num(
        train_stat, nan=0.0, posinf=1e12, neginf=-1e12
    ))
    refs = {}
    for c in range(k):
        vals_c = train_stat[train_pred == c]
        vals_c = vals_c[np.isfinite(vals_c)]
        if vals_c.size >= 12:
            refs[c] = np.sort(vals_c)

    out = np.empty(x_eval.shape[0], dtype=float)
    for i, (value, c) in enumerate(zip(eval_stat, eval_pred)):
        ref = refs.get(int(c), global_ref)
        value = float(np.nan_to_num(value, nan=0.0, posinf=1e12, neginf=-1e12))
        rank = np.searchsorted(ref, value, side="right") / float(ref.size)
        scale = np.std(ref) + 1e-8
        out[i] = rank + 1e-5 * (value - np.median(ref)) / scale

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
