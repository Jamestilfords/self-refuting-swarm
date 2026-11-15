import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "After projecting out the class-mean discriminative subspace, in-scope examples should retain "
    "locally supported residual variation. Class-scaled Euclidean kNN distance in this residual space "
    "flags novel semantic components while preserving residual magnitude information."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0207", "m_0272"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n_classes = tr.logits.shape[1]
    global_mean = np.mean(x, axis=0)
    class_means = np.empty((n_classes, x.shape[1]), dtype=float)
    for c in range(n_classes):
        mask = y == c
        class_means[c] = np.mean(x[mask], axis=0) if np.any(mask) else global_mean

    mean_offsets = class_means - global_mean
    _, singular, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    if singular.size:
        threshold = max(float(singular[0]) * 1e-8, 1e-12)
        rank = int(np.sum(singular > threshold))
    else:
        rank = 0

    if rank:
        basis = vt[:rank]
        xc = x - global_mean
        zc = z - global_mean
        xr = xc - (xc @ basis.T) @ basis
        zr = zc - (zc @ basis.T) @ basis
    else:
        xr = x - global_mean
        zr = z - global_mean

    global_scale = np.sqrt(np.mean(xr * xr) + 1e-12)
    scales = np.full(n_classes, global_scale, dtype=float)
    for c in range(n_classes):
        mask = y == c
        if np.any(mask):
            scales[c] = max(np.sqrt(np.mean(xr[mask] * xr[mask]) + 1e-12),
                            global_scale * 1e-3)

    pred = np.argmax(ev.logits, axis=1)
    train_scale = scales[np.clip(y, 0, n_classes - 1)]
    eval_scale = scales[np.clip(pred, 0, n_classes - 1)]

    xr = xr / train_scale[:, None]
    zr = zr / eval_scale[:, None]
    return np.asarray(ctx.h.knn(xr, zr, k=10, l2=False), dtype=float)
