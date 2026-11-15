import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class centroids span directions that most distinguish known intents in the bottleneck representation. Queries with unusually large energy outside this discriminative class-mean subspace are scored as more likely out-of-scope."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels)

    classes = np.unique(labels)
    mean = x_train.mean(axis=0)
    class_means = np.vstack([
        x_train[labels == c].mean(axis=0) - mean for c in classes
    ])

    if class_means.shape[0] > 1:
        _, singular_values, vt = np.linalg.svd(class_means, full_matrices=False)
        rank = int(np.sum(singular_values > 1e-10))
        rank = max(1, min(rank, 64, vt.shape[0]))
        basis = vt[:rank].T
        centered_train = x_train - mean
        centered_eval = x_eval - mean
        train_residual = centered_train - (centered_train @ basis) @ basis.T
        eval_residual = centered_eval - (centered_eval @ basis) @ basis.T
    else:
        train_residual = x_train - mean
        eval_residual = x_eval - mean

    train_score = np.sum(train_residual * train_residual, axis=1)
    scale = np.median(train_score)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.mean(train_score) + 1e-12

    residual_score = np.sum(eval_residual * eval_residual, axis=1) / scale
    total_energy = np.sum((x_eval - mean) ** 2, axis=1) / (np.mean(np.sum((x_train - mean) ** 2, axis=1)) + 1e-12)
    return residual_score + 1e-6 * total_energy
