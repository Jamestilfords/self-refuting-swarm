import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class means define directions used to distinguish known intents, while variation orthogonal to that subspace is less explained by the intent taxonomy. Queries with unusually large residual energy are therefore more likely to be out-of-scope."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    classes, inv = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    counts = np.bincount(inv, minlength=n_classes).astype(float)

    mean = x.mean(axis=0)
    class_sums = np.zeros((n_classes, x.shape[1]), dtype=float)
    np.add.at(class_sums, inv, x)
    class_means = class_sums / np.maximum(counts[:, None], 1.0)

    mean_offsets = (class_means - mean) * np.sqrt(counts[:, None])
    _, singular_values, vt = np.linalg.svd(mean_offsets, full_matrices=False)

    energy = singular_values ** 2
    total_energy = float(energy.sum())
    if total_energy > 1e-12:
        cumulative = np.cumsum(energy) / total_energy
        rank = int(np.searchsorted(cumulative, 0.90) + 1)
    else:
        rank = 1
    rank = max(1, min(rank, 64, vt.shape[0]))
    basis = vt[:rank].T

    train_centered = x - mean
    train_projection = train_centered @ basis
    train_residual_sq = np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(train_projection * train_projection, axis=1),
        0.0,
    )

    residual_center = float(np.median(train_residual_sq))
    residual_scale = float(
        np.median(np.abs(train_residual_sq - residual_center))
    )
    residual_scale = max(1.4826 * residual_scale, 1e-8)

    eval_centered = np.asarray(ev.pooled, dtype=float) - mean
    eval_projection = eval_centered @ basis
    eval_residual_sq = np.maximum(
        np.sum(eval_centered * eval_centered, axis=1)
        - np.sum(eval_projection * eval_projection, axis=1),
        0.0,
    )

    return (eval_residual_sq - residual_center) / residual_scale
