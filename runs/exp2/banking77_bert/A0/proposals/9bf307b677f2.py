import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-mean directions capture variation that separates known intents, while the orthogonal remainder captures within-intent constraints that OOS queries need not satisfy. Calibrating residual energy by predicted intent accounts for classes with different residual variability."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0087"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    global_mean = x.mean(axis=0)
    means = np.zeros((k, d), dtype=float)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)
    means[counts == 0] = global_mean

    between = means - global_mean
    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    if singular.size:
        tol = singular[0] * max(between.shape) * np.finfo(float).eps
        rank = max(1, int(np.sum(singular > tol)))
        basis = vt[:rank].T
    else:
        basis = np.zeros((d, 0), dtype=float)

    def residual_energy(z):
        centered = z - global_mean
        total = np.sum(centered * centered, axis=1)
        if basis.shape[1]:
            projected = centered @ basis
            total = total - np.sum(projected * projected, axis=1)
        return np.maximum(total, 0.0)

    train_r = np.log1p(residual_energy(x))
    class_loc = np.zeros(k, dtype=float)
    class_scale = np.zeros(k, dtype=float)
    for c in range(k):
        values = train_r[y == c]
        if values.size:
            class_loc[c] = np.median(values)
            class_scale[c] = np.median(np.abs(values - class_loc[c])) * 1.4826

    global_scale = np.median(np.abs(train_r - np.median(train_r))) * 1.4826
    global_scale = max(float(global_scale), 1e-6)
    class_scale = np.maximum(class_scale, global_scale * 0.1)

    pred = np.argmax(ev.logits, axis=1)
    eval_r = np.log1p(residual_energy(np.asarray(ev.feats, dtype=float)))
    return (eval_r - class_loc[pred]) / class_scale[pred]
