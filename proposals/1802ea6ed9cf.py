import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class-mean directions capture the representation variation most useful for separating known intents. "
    "After removing them, an in-scope query should still lie near the low-rank residual manifold of its classifier-selected intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0207"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    y_train = np.asarray(tr.labels, dtype=int)

    grand = np.mean(x_train, axis=0)
    classes = np.unique(y_train)
    means = np.vstack([np.mean(x_train[y_train == c], axis=0) for c in classes])
    mean_offsets = means - grand

    _, singular, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    if singular.size:
        cutoff = max(float(singular[0]) * 1e-8, 1e-10)
        rank = int(np.sum(singular > cutoff))
    else:
        rank = 0
    u = vt[:rank].T if rank > 0 else np.empty((x_train.shape[1], 0))

    def residual(x):
        z = x - grand
        if rank:
            z = z - (z @ u) @ u.T
        return z

    r_train = residual(x_train)
    r_eval = residual(x_eval)
    dim = r_train.shape[1]

    def fit_model(z):
        center = np.mean(z, axis=0)
        centered = z - center
        n = centered.shape[0]
        if n >= 4 and dim > 1:
            _, s, v = np.linalg.svd(centered, full_matrices=False)
            keep = min(12, max(1, n - 3), v.shape[0])
            if s.size:
                keep = min(keep, max(1, int(np.sum(s > max(s[0] * 1e-8, 1e-10)))))
            basis = v[:keep].T
        else:
            basis = np.empty((dim, 0))

        total = np.sum(centered * centered, axis=1)
        if basis.shape[1]:
            explained = np.sum((centered @ basis) ** 2, axis=1)
        else:
            explained = np.zeros(n)
        errors = np.maximum(total - explained, 0.0)

        scale = float(np.median(errors) + np.mean(errors) * 0.1 + 1e-8)
        train_raw = np.log1p(errors / scale)
        return center, basis, scale, np.sort(train_raw)

    global_model = fit_model(r_train)
    models = {}
    for c in classes:
        zc = r_train[y_train == c]
        models[int(c)] = fit_model(zc) if zc.shape[0] >= 3 else global_model

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    out = np.empty(x_eval.shape[0], dtype=float)

    for c in np.unique(pred):
        idx = np.flatnonzero(pred == c)
        center, basis, scale, reference = models.get(int(c), global_model)
        z = r_eval[idx] - center
        total = np.sum(z * z, axis=1)
        if basis.shape[1]:
            explained = np.sum((z @ basis) ** 2, axis=1)
        else:
            explained = np.zeros(z.shape[0])
        errors = np.maximum(total - explained, 0.0)
        raw = np.log1p(errors / scale)

        tail = np.searchsorted(reference, raw, side="right") / float(reference.size)
        out[idx] = tail + 1e-3 * raw

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
