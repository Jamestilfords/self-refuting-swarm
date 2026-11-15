import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent should be supported under that intent's own coordinate-wise variance structure. "
    "Shrinking per-class diagonal variances toward global variances stabilizes rare classes while preserving tightly constrained intent dimensions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0216", "m_0094", "m_0013"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)

    global_mean = np.mean(x_train, axis=0)
    global_var = np.mean((x_train - global_mean) ** 2, axis=0) + 1e-6

    means = np.empty((n_classes, dim), dtype=np.float64)
    variances = np.empty((n_classes, dim), dtype=np.float64)
    offsets = np.empty(n_classes, dtype=np.float64)
    scales = np.empty(n_classes, dtype=np.float64)

    for c in range(n_classes):
        members = x_train[labels == c]
        count = len(members)
        if count == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            mean = np.mean(members, axis=0)
            raw_var = np.mean((members - mean) ** 2, axis=0)
            shrink = dim / (dim + count)
            means[c] = mean
            variances[c] = (1.0 - shrink) * raw_var + shrink * global_var + 1e-6

        own = x_train[pred_train == c]
        if len(own) == 0:
            own = x_train[labels == c]
        if len(own) == 0:
            offsets[c] = 0.0
            scales[c] = 1.0
        else:
            z = (own - means[c]) ** 2 / variances[c]
            radii = np.mean(z, axis=1)
            offsets[c] = np.median(radii)
            scales[c] = np.median(np.abs(radii - offsets[c])) + 1e-6

    diff = x_eval - means[pred_eval]
    radius = np.mean(diff * diff / variances[pred_eval], axis=1)
    calibrated = (radius - offsets[pred_eval]) / scales[pred_eval]
    return np.nan_to_num(calibrated, nan=0.0, posinf=1e6, neginf=-1e6)
