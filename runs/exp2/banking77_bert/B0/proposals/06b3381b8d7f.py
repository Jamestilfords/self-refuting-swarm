import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Within-class trailing covariance directions capture tightly constrained in-scope variation that broad principal directions can hide. A predicted-class, regularized trailing-subspace residual flags queries that violate those low-variance intent-specific constraints."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]
    d = x.shape[1]
    tail_dim = min(128, max(16, d // 6))

    counts = np.bincount(y, minlength=k).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)
    np.add.at(sums, y, x)
    means = sums / np.maximum(counts[:, None], 1.0)

    residuals = x - means[y]
    cov = (residuals.T @ residuals) / max(len(x) - k, 1)
    cov.flat[::d + 1] += 1e-6 * (np.trace(cov) / max(d, 1) + 1.0)

    vals, vecs = np.linalg.eigh(cov)
    basis = vecs[:, :tail_dim]
    tail_vals = np.maximum(vals[:tail_dim], 0.0)
    floor = max(float(vals[tail_dim - 1]) if tail_dim > 0 else 0.0, 1e-8)
    denom = np.maximum(tail_vals, floor * 0.05)

    train_proj = residuals @ basis
    train_q = np.sum((train_proj * train_proj) / denom[None, :], axis=1)
    train_s = np.log1p(train_q)

    class_sum = np.bincount(y, weights=train_s, minlength=k)
    class_mean = class_sum / np.maximum(counts, 1.0)
    centered = train_s - class_mean[y]
    class_var = np.bincount(y, weights=centered * centered, minlength=k)
    class_std = np.sqrt(class_var / np.maximum(counts - 1.0, 1.0))
    global_std = max(float(np.std(train_s)), 1e-6)
    class_std = np.maximum(class_std, 0.25 * global_std)

    pred = np.argmax(ev.logits, axis=1)
    eval_residuals = z - means[pred]
    eval_proj = eval_residuals @ basis
    eval_q = np.sum((eval_proj * eval_proj) / denom[None, :], axis=1)
    scores = (np.log1p(eval_q) - class_mean[pred]) / class_std[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
