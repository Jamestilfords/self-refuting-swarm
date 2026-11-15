import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode constraints that in-scope utterances rarely violate. "
    "A maximum standardized deviation over these trailing directions detects localized constraint violations that can be diluted by summed Mahalanobis energy."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0135"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    xe = np.asarray(ev.pooled, dtype=np.float64)
    k = tr.logits.shape[1]
    d = x.shape[1]

    class_means = np.zeros((k, d), dtype=np.float64)
    counts = np.bincount(y, minlength=k).astype(np.float64)
    global_mean = x.mean(axis=0)
    for c in range(k):
        if counts[c] > 0:
            class_means[c] = x[y == c].mean(axis=0)
        else:
            class_means[c] = global_mean

    residuals = x - class_means[y]
    cov = (residuals.T @ residuals) / max(len(x) - k, 1)
    ridge = max(float(np.trace(cov)) / max(d, 1), 1e-8) * 1e-6
    vals, vecs = np.linalg.eigh(cov + ridge * np.eye(d))

    n_tail = min(48, d)
    tail = vecs[:, :n_tail]
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    proj_train = (x - class_means[train_pred]) @ tail
    proj_eval = (xe - class_means[eval_pred]) @ tail

    global_scale = np.sqrt(np.mean(proj_train * proj_train, axis=0) + 1e-10)
    scale = np.empty((k, n_tail), dtype=np.float64)
    for c in range(k):
        mask = train_pred == c
        nc = int(mask.sum())
        if nc >= 8:
            local_var = np.mean(proj_train[mask] * proj_train[mask], axis=0)
            weight = nc / (nc + 32.0)
            scale[c] = np.sqrt(weight * local_var + (1.0 - weight) * global_scale**2 + 1e-10)
        else:
            scale[c] = global_scale

    raw_train = np.max(np.abs(proj_train) / scale[train_pred], axis=1)
    raw_eval = np.max(np.abs(proj_eval) / scale[eval_pred], axis=1)

    center = np.zeros(k, dtype=np.float64)
    spread = np.ones(k, dtype=np.float64)
    global_center = float(np.median(raw_train))
    global_spread = float(np.std(raw_train) + 1e-6)
    for c in range(k):
        mask = train_pred == c
        nc = int(mask.sum())
        if nc >= 8:
            local_center = float(np.median(raw_train[mask]))
            local_spread = float(np.std(raw_train[mask]) + 1e-6)
            weight = nc / (nc + 32.0)
            center[c] = weight * local_center + (1.0 - weight) * global_center
            spread[c] = weight * local_spread + (1.0 - weight) * global_spread
        else:
            center[c] = global_center
            spread[c] = global_spread

    return (raw_eval - center[eval_pred]) / spread[eval_pred]
