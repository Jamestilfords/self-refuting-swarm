import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean subspace suppresses directions used to separate known intents, "
    "so nearest-neighbor novelty in the residual representation highlights queries with unfamiliar non-intent structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels)

    global_mean = np.mean(x_train, axis=0)
    classes = np.unique(labels)
    class_means = np.vstack([
        np.mean(x_train[labels == c], axis=0) for c in classes
    ])
    offsets = class_means - global_mean

    _, singular_values, vt = np.linalg.svd(offsets, full_matrices=False)
    if singular_values.size:
        cutoff = max(float(singular_values[0]) * 1e-7, 1e-12)
        rank = int(np.sum(singular_values > cutoff))
        rank = max(1, min(rank, vt.shape[0], 96))
        basis = vt[:rank].T
    else:
        basis = np.zeros((x_train.shape[1], 0), dtype=np.float64)

    train_centered = x_train - global_mean
    eval_centered = x_eval - global_mean

    if basis.shape[1]:
        train_residual = train_centered - (train_centered @ basis) @ basis.T
        eval_residual = eval_centered - (eval_centered @ basis) @ basis.T
    else:
        train_residual = train_centered
        eval_residual = eval_centered

    train_energy = np.sum(train_residual * train_residual, axis=1)
    eval_energy = np.sum(eval_residual * eval_residual, axis=1)
    energy_scale = float(np.median(train_energy))
    if not np.isfinite(energy_scale) or energy_scale <= 1e-12:
        energy_scale = float(np.mean(train_energy) + 1e-12)

    k = max(1, min(15, train_residual.shape[0]))
    residual_knn = ctx.h.knn(train_residual, eval_residual, k=k, l2=True)
    radial_novelty = np.log1p(np.maximum(eval_energy, 0.0) / energy_scale)

    scores = np.asarray(residual_knn, dtype=np.float64) + 0.12 * radial_novelty
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
