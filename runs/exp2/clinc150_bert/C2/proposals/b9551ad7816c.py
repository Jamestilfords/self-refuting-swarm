import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean directions capture the discriminative subspace used to separate known intents. Queries with unusually large residual energy orthogonal to that subspace are less consistent with in-scope variation."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    global_mean = x_train.mean(axis=0)
    means = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(means, labels, x_train)
    present = counts > 0
    means[present] /= counts[present, None]
    means[~present] = global_mean

    centered_means = means[present] - global_mean
    weights = np.sqrt(counts[present])[:, None]
    basis_source = centered_means * weights
    _, singular_values, vt = np.linalg.svd(basis_source, full_matrices=False)
    if singular_values.size:
        tol = singular_values[0] * 1e-10
        rank = max(1, int(np.sum(singular_values > tol)))
        basis = vt[:rank].T
        train_centered = x_train - global_mean
        eval_centered = x_eval - global_mean
        train_resid = train_centered - (train_centered @ basis) @ basis.T
        eval_resid = eval_centered - (eval_centered @ basis) @ basis.T
    else:
        train_resid = x_train - global_mean
        eval_resid = x_eval - global_mean

    train_energy = np.sum(train_resid * train_resid, axis=1)
    eval_energy = np.sum(eval_resid * eval_resid, axis=1)

    class_median = np.full(n_classes, np.median(train_energy), dtype=float)
    class_scale = np.full(n_classes, np.std(train_energy) + 1e-6, dtype=float)
    for c in np.flatnonzero(present):
        values = train_energy[labels == c]
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        class_median[c] = med
        class_scale[c] = max(1.4826 * mad, np.std(values) * 0.1, 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    scores = (eval_energy - class_median[predicted]) / class_scale[predicted]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
