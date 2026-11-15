import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean directions capture feature variation that discriminates known intents, while the orthogonal complement contains semantic variation not explained by any learned intent separation. OOS queries should retain larger residual energy after removal of this supervised discriminative subspace."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0005"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    means = np.zeros((n_classes, x.shape[1]), dtype=np.float64)
    np.add.at(means, y, x)
    means /= np.maximum(counts[:, None], 1.0)

    global_mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.maximum(scale, 1e-6)

    active = counts > 0
    centered_means = (means[active] - global_mean) / scale
    weighted_means = centered_means * np.sqrt(counts[active, None] / counts[active].sum())

    _, singular_values, vt = np.linalg.svd(weighted_means, full_matrices=False)
    if singular_values.size:
        keep = singular_values > (singular_values[0] * 1e-8)
    else:
        keep = np.zeros(0, dtype=bool)

    standardized = (z - global_mean) / scale
    if np.any(keep):
        basis = vt[keep].T
        projection = standardized @ basis
        residual_sq = np.sum(standardized * standardized, axis=1) - np.sum(projection * projection, axis=1)
    else:
        residual_sq = np.sum(standardized * standardized, axis=1)

    return np.maximum(np.nan_to_num(residual_sq, nan=0.0, posinf=1e12, neginf=0.0), 0.0)
