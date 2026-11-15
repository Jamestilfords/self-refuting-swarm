import numpy as np
from sklearn.decomposition import PCA

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Each predicted intent has its own low-variance within-class directions after a shared "
    "semantic projection. OOS queries should exhibit unusually large standardized residuals "
    "along these tightly constrained directions even when broad semantic variation looks familiar."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0053"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    y_train = np.asarray(tr.labels, dtype=np.int64)

    n, d = x_train.shape
    k = tr.logits.shape[1]
    dim = max(2, min(128, d, n - 1))

    pca = PCA(n_components=dim, svd_solver="randomized", random_state=2718)
    z_train = pca.fit_transform(x_train)
    z_eval = pca.transform(x_eval)

    global_mean = z_train.mean(axis=0)
    centered = z_train - global_mean
    global_cov = centered.T @ centered / max(n - 1, 1)
    global_vals, global_vecs = np.linalg.eigh(global_cov)
    global_vals = np.maximum(global_vals, 1e-6 * max(float(global_vals.max()), 1.0))

    tail_dim = max(2, min(24, dim // 3))
    means = np.empty((k, dim), dtype=np.float64)
    tail_vecs = np.empty((k, dim, tail_dim), dtype=np.float64)
    tail_vars = np.empty((k, tail_dim), dtype=np.float64)

    for c in range(k):
        members = z_train[y_train == c]
        if len(members) >= max(6, tail_dim + 2):
            mean = members.mean(axis=0)
            residual = members - mean
            cov = residual.T @ residual / max(len(members) - 1, 1)
            vals, vecs = np.linalg.eigh(cov)
            floor = 1e-4 * max(float(vals.max()), 1.0)
            vals = np.maximum(vals, floor)
        else:
            mean = global_mean
            vals = global_vals
            vecs = global_vecs

        means[c] = mean
        tail_vecs[c] = vecs[:, :tail_dim]
        tail_vars[c] = vals[:tail_dim]

    train_pred = np.argmax(tr.logits, axis=1)
    train_raw = np.empty(n, dtype=np.float64)
    for c in range(k):
        idx = np.flatnonzero(train_pred == c)
        if idx.size:
            residual = z_train[idx] - means[c]
            projected = residual @ tail_vecs[c]
            train_raw[idx] = np.sum((projected * projected) / tail_vars[c], axis=1)

    pooled_sorted = np.sort(train_raw)
    per_class_sorted = []
    for c in range(k):
        values = train_raw[train_pred == c]
        per_class_sorted.append(np.sort(values) if values.size >= 12 else pooled_sorted)

    eval_pred = np.argmax(ev.logits, axis=1)
    out = np.empty(len(z_eval), dtype=np.float64)
    for c in range(k):
        idx = np.flatnonzero(eval_pred == c)
        if idx.size:
            residual = z_eval[idx] - means[c]
            projected = residual @ tail_vecs[c]
            raw = np.sum((projected * projected) / tail_vars[c], axis=1)
            reference = per_class_sorted[c]
            rank = np.searchsorted(reference, raw, side="right") / (len(reference) + 1.0)
            out[idx] = np.log1p(raw) + rank

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
