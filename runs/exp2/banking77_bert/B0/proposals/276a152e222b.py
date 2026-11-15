import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "The class-mean subspace contains feature directions that separate known intents, so scoring the orthogonal remainder "
    "with predicted-class residual variances detects inputs that look discriminative but violate intent-specific nuisance structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    q = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    mu = np.mean(x, axis=0)
    valid = (y >= 0) & (y < k)
    counts = np.bincount(y[valid], minlength=k).astype(float)
    means = np.zeros((k, d), dtype=float)
    np.add.at(means, y[valid], x[valid])
    nonempty = counts > 0
    means[nonempty] /= counts[nonempty, None]
    means -= mu

    _, singular, vt = np.linalg.svd(means[nonempty], full_matrices=False)
    if singular.size:
        threshold = max(float(singular[0]) * 1e-6, 1e-10)
        rank = int(np.sum(singular > threshold))
    else:
        rank = 0

    if rank >= d:
        rank = d - 1
    if rank > 0:
        basis = vt[:rank]
        xr = (x - mu) - ((x - mu) @ basis.T) @ basis
        qr = (q - mu) - ((q - mu) @ basis.T) @ basis
    else:
        xr = x - mu
        qr = q - mu

    global_var = np.mean(xr * xr, axis=0)
    floor = max(float(np.median(global_var)) * 1e-4, 1e-8)
    global_var = np.maximum(global_var, floor)

    class_var = np.empty((k, d), dtype=float)
    for c in range(k):
        if counts[c] >= 2:
            rc = xr[y == c]
            v = np.mean((rc - np.mean(rc, axis=0)) ** 2, axis=0)
            class_var[c] = np.maximum(v, floor)
        else:
            class_var[c] = global_var

    pred = np.argmax(ev.logits, axis=1)
    residual_energy = np.sum((qr * qr) / class_var[pred], axis=1)
    train_pred = np.argmax(tr.logits, axis=1)
    train_energy = np.sum((xr * xr) / class_var[train_pred], axis=1)

    center = np.median(train_energy)
    scale = np.median(np.abs(train_energy - center)) * 1.4826
    scale = max(float(scale), 1e-8)
    return np.clip((residual_energy - center) / scale, -1e6, 1e6)
