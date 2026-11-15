import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries may have atypical residuals specifically in low-variance within-class directions. This score fits those trailing directions on centered in-scope features and measures the nearest class-conditional trailing-space deviation."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels)
    classes = np.unique(labels)

    global_mean = x.mean(axis=0)
    class_means = np.empty((len(classes), x.shape[1]), dtype=np.float64)
    centered = np.empty_like(x)

    for j, c in enumerate(classes):
        mask = labels == c
        mu = x[mask].mean(axis=0)
        class_means[j] = mu
        centered[mask] = x[mask] - mu

    cov = centered.T @ centered / max(len(x) - len(classes), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    n_tail = min(64, x.shape[1])
    tail_vals = eigvals[:n_tail]
    tail_vecs = eigvecs[:, :n_tail]
    scale_floor = max(float(np.median(eigvals)), 1e-8) * 1e-3
    inv_scale = 1.0 / np.sqrt(np.maximum(tail_vals, scale_floor))

    train_tail_means = class_means @ tail_vecs
    eval_tail = z @ tail_vecs
    eval_scaled = eval_tail * inv_scale
    mean_scaled = train_tail_means * inv_scale

    eval_norm = np.sum(eval_scaled * eval_scaled, axis=1, keepdims=True)
    mean_norm = np.sum(mean_scaled * mean_scaled, axis=1)
    distances = eval_norm + mean_norm[None, :] - 2.0 * (eval_scaled @ mean_scaled.T)
    scores = np.min(distances, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
