import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Removing the class-mean subspace isolates feature variation not needed to distinguish known intents. Class-specific diagonal precision in this discriminative complement measures whether a query violates the residual semantic tolerances of the intent selected by its logits."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0051"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes = np.unique(y)
    dim = x.shape[1]
    global_mean = np.mean(x, axis=0)
    class_means = np.vstack([np.mean(x[y == c], axis=0) for c in classes])
    between = class_means - np.mean(class_means, axis=0, keepdims=True)

    _, singular, vt = np.linalg.svd(between, full_matrices=False)
    if singular.size:
        threshold = singular[0] * max(between.shape) * np.finfo(float).eps
        rank = int(np.sum(singular > threshold))
    else:
        rank = 0
    basis = vt[:rank].T if rank else np.empty((dim, 0), dtype=float)

    xc = x - global_mean
    zc = z - global_mean
    if rank:
        xr = xc - (xc @ basis) @ basis.T
        zr = zc - (zc @ basis) @ basis.T
    else:
        xr = xc
        zr = zc

    predicted_train = np.argmax(tr.logits, axis=1)
    predicted_eval = np.argmax(ev.logits, axis=1)
    global_var = np.var(xr, axis=0) + 1e-8
    variances = np.empty((tr.logits.shape[1], dim), dtype=float)

    for c in range(variances.shape[0]):
        members = xr[predicted_train == c]
        if members.shape[0] >= 4:
            local_var = np.var(members, axis=0)
            weight = members.shape[0] / (members.shape[0] + 20.0)
            variances[c] = weight * local_var + (1.0 - weight) * global_var
        else:
            variances[c] = global_var

    precision = 1.0 / np.maximum(variances, 1e-8)
    scores = np.sum(zr * zr * precision[predicted_eval], axis=1)
    return np.log1p(np.maximum(scores, 0.0))
