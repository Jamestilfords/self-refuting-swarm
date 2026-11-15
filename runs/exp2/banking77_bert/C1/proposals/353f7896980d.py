import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Class-mean directions encode the dominant discriminative structure for known intents. "
    "Local neighbor distance in the normalized orthogonal residual space detects queries with unfamiliar non-discriminative representation patterns."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels)

    classes, inv = np.unique(y, return_inverse=True)
    n_classes = classes.size
    counts = np.bincount(inv, minlength=n_classes).astype(float)

    global_mean = np.mean(x, axis=0)
    sums = np.zeros((n_classes, x.shape[1]), dtype=float)
    np.add.at(sums, inv, x)
    class_means = sums / np.maximum(counts[:, None], 1.0)

    between = (class_means - global_mean) * np.sqrt(
        counts[:, None] / max(float(counts.sum()), 1.0)
    )
    _, singular_values, vt = np.linalg.svd(between, full_matrices=False)

    if singular_values.size:
        threshold = max(float(singular_values[0]) * 1e-8, 1e-12)
        available = int(np.sum(singular_values > threshold))
    else:
        available = 0
    rank = max(1, min(available, 64, vt.shape[0]))
    basis = vt[:rank]

    xc = x - global_mean
    zc = z - global_mean
    train_residual = xc - (xc @ basis.T) @ basis
    eval_residual = zc - (zc @ basis.T) @ basis

    k = min(17, max(1, train_residual.shape[0]))
    distances = ctx.h.knn(train_residual, eval_residual, k=k, l2=True)
    return np.nan_to_num(
        np.asarray(distances, dtype=float),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )
