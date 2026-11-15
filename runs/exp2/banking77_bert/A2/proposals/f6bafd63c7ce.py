import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean subspace suppresses directions used primarily to distinguish known intents. "
    "OOS inputs should show unusually large class-conditioned deviation in the remaining residual geometry."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0067"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n_classes = tr.logits.shape[1]
    dim = x.shape[1]
    global_mean = x.mean(axis=0)

    counts = np.bincount(y, minlength=n_classes).astype(float)
    means = np.tile(global_mean, (n_classes, 1))
    np.add.at(means, y, x)
    nonempty = counts > 0
    means[nonempty] /= counts[nonempty, None]

    mean_offsets = means[nonempty] - global_mean
    if mean_offsets.shape[0] > 1:
        _, singular, vt = np.linalg.svd(mean_offsets, full_matrices=False)
        rank = min(
            vt.shape[0],
            max(1, min(n_classes - 1, int(np.sum(singular > singular[0] * 1e-8))))
        )
        basis = vt[:rank]
        xc = x - global_mean
        zc = z - global_mean
        xr = xc - (xc @ basis.T) @ basis
        zr = zc - (zc @ basis.T) @ basis
    else:
        xr = x - global_mean
        zr = z - global_mean

    residual_means = np.zeros((n_classes, dim), dtype=float)
    np.add.at(residual_means, y, xr)
    residual_means[nonempty] /= counts[nonempty, None]

    centered = xr - residual_means[y]
    global_var = np.mean(centered * centered, axis=0) + 1e-8

    class_ss = np.zeros((n_classes, dim), dtype=float)
    np.add.at(class_ss, y, centered * centered)
    class_var = class_ss / np.maximum(counts[:, None], 1.0)

    shrink = counts[:, None] / (counts[:, None] + 16.0)
    variances = shrink * class_var + (1.0 - shrink) * global_var[None, :]
    variances = np.maximum(variances, 1e-8)

    pred = np.argmax(ev.logits, axis=1)
    deviation = zr - residual_means[pred]
    return np.mean((deviation * deviation) / variances[pred], axis=1)
