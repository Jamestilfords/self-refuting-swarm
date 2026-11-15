import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "OOS inputs can produce Dirichlet concentration profiles whose relative allocation across classes differs from the class-specific profiles seen in training, even when total evidence is similar. This score removes global concentration scale and measures a predicted-class conditional anomaly in centered log-concentration shape."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    z_tr = np.log(a_tr)
    z_ev = np.log(a_ev)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    n, k = z_tr.shape
    labels = tr.labels.astype(int)
    counts = np.bincount(labels, minlength=k).astype(float)
    counts = np.maximum(counts, 1.0)

    sums = np.zeros((k, k), dtype=float)
    sq_sums = np.zeros((k, k), dtype=float)
    np.add.at(sums, labels, z_tr)
    np.add.at(sq_sums, labels, z_tr * z_tr)

    means = sums / counts[:, None]
    variances = sq_sums / counts[:, None] - means * means

    global_var = np.var(z_tr, axis=0)
    floor = np.maximum(1e-6, 0.02 * global_var)
    variances = np.maximum(variances, floor[None, :])

    predicted = np.argmax(a_ev, axis=1)
    delta = z_ev - means[predicted]
    shape_distance = np.mean(
        (delta * delta) / variances[predicted],
        axis=1,
    )

    return np.nan_to_num(shape_distance, nan=0.0, posinf=1e12, neginf=0.0)
