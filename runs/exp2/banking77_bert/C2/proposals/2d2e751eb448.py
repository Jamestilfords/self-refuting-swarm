import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-mean directions capture discriminative in-scope variation, while OOS queries should have unusually large residual distance from their predicted class in the orthogonal remainder. Class-specific residual calibration avoids penalizing naturally diffuse intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=float)
    z = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    n_classes = int(np.max(y)) + 1
    dim = x.shape[1]
    global_mean = x.mean(axis=0)

    class_means = np.empty((n_classes, dim), dtype=float)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    for c in range(n_classes):
        if counts[c] > 0:
            class_means[c] = x[y == c].mean(axis=0)
        else:
            class_means[c] = global_mean

    mean_offsets = class_means - global_mean
    _, singular, vt = np.linalg.svd(mean_offsets, full_matrices=False)
    rank = min(128, np.sum(singular > 1e-10))
    if rank > 0:
        basis = vt[:rank].T
        train_centered = x - global_mean
        eval_centered = z - global_mean
        train_residual = train_centered - (train_centered @ basis) @ basis.T
        eval_residual = eval_centered - (eval_centered @ basis) @ basis.T
        class_residual_means = mean_offsets - (mean_offsets @ basis) @ basis.T
    else:
        train_residual = x - global_mean
        eval_residual = z - global_mean
        class_residual_means = mean_offsets

    train_delta = train_residual - class_residual_means[y]
    train_d2 = np.sum(train_delta * train_delta, axis=1)

    class_loc = np.zeros(n_classes, dtype=float)
    class_scale = np.ones(n_classes, dtype=float)
    for c in range(n_classes):
        values = train_d2[y == c]
        if values.size:
            class_loc[c] = np.median(values)
            mad = np.median(np.abs(values - class_loc[c]))
            class_scale[c] = max(1e-8, 1.4826 * mad)

    pred = np.argmax(ev.logits, axis=1)
    pred = np.clip(pred, 0, n_classes - 1)
    eval_delta = eval_residual - class_residual_means[pred]
    eval_d2 = np.sum(eval_delta * eval_delta, axis=1)

    return (eval_d2 - class_loc[pred]) / class_scale[pred]
