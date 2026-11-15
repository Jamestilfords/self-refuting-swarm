import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-mean directions capture discriminative intent variation, while the orthogonal residual retains nuisance and off-manifold variation. A query whose residual is far from all training residuals is likely out-of-scope even when its class-direction projection appears plausible."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.nan_to_num(
        np.asarray(tr.feats, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )
    x_eval = np.nan_to_num(
        np.asarray(ev.feats, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )
    labels = np.asarray(tr.labels)

    global_mean = x_train.mean(axis=0)
    classes = np.unique(labels)
    class_offsets = np.vstack(
        [x_train[labels == c].mean(axis=0) - global_mean for c in classes]
    )

    if class_offsets.shape[0] > 1:
        _, singular, vt = np.linalg.svd(class_offsets, full_matrices=False)
        threshold = (
            max(class_offsets.shape)
            * np.finfo(float).eps
            * (singular[0] if singular.size else 1.0)
        )
        rank = int(np.sum(singular > threshold))
        basis = vt[:rank].T
    else:
        basis = np.empty((x_train.shape[1], 0), dtype=float)

    train_centered = x_train - global_mean
    eval_centered = x_eval - global_mean

    if basis.shape[1]:
        train_residual = train_centered - (train_centered @ basis) @ basis.T
        eval_residual = eval_centered - (eval_centered @ basis) @ basis.T
    else:
        train_residual = train_centered
        eval_residual = eval_centered

    residual_knn = ctx.h.knn(train_residual, eval_residual, k=10, l2=False)
    residual_energy = np.sum(eval_residual * eval_residual, axis=1)
    scale = np.median(np.sum(train_residual * train_residual, axis=1)) + 1e-12

    scores = np.asarray(residual_knn, dtype=float) + 1e-3 * residual_energy / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
