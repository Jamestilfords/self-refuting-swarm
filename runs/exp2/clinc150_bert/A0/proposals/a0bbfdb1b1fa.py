import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Removing the class-mean discriminative subspace isolates feature variation not needed to distinguish known intents. "
    "A predicted-intent conditional, shrunk diagonal likelihood in this orthogonal remainder flags inputs with unsupported residual structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0021", "m_0023"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    global_mean = np.mean(x_train, axis=0)
    centered_train = x_train - global_mean
    centered_eval = x_eval - global_mean

    classes = np.unique(labels)
    class_offsets = np.vstack([
        np.mean(x_train[labels == c], axis=0) - global_mean
        for c in classes
    ])

    if class_offsets.shape[0] > 0:
        _, singular_values, vt = np.linalg.svd(class_offsets, full_matrices=False)
        if singular_values.size and singular_values[0] > 0.0:
            tol = singular_values[0] * max(class_offsets.shape) * np.finfo(np.float64).eps
            rank = int(np.sum(singular_values > tol))
        else:
            rank = 0
    else:
        vt = np.empty((0, x_train.shape[1]), dtype=np.float64)
        rank = 0

    if rank > 0:
        basis = vt[:rank].T
        train_residual = centered_train - (centered_train @ basis) @ basis.T
        eval_residual = centered_eval - (centered_eval @ basis) @ basis.T
    else:
        train_residual = centered_train
        eval_residual = centered_eval

    global_res_mean = np.mean(train_residual, axis=0)
    global_var = np.var(train_residual, axis=0)
    positive_var = global_var[global_var > 0.0]
    floor = (np.median(positive_var) * 1e-5) if positive_var.size else 1e-8
    floor = max(float(floor), 1e-10)
    global_var = np.maximum(global_var, floor)

    n_classes = tr.logits.shape[1]
    means = np.tile(global_res_mean, (n_classes, 1))
    variances = np.tile(global_var, (n_classes, 1))

    for c in classes:
        if c < 0 or c >= n_classes:
            continue
        residuals = train_residual[labels == c]
        count = residuals.shape[0]
        if count == 0:
            continue
        class_mean = np.mean(residuals, axis=0)
        class_var = np.var(residuals, axis=0)
        shrink = count / (count + 20.0)
        means[c] = shrink * class_mean + (1.0 - shrink) * global_res_mean
        variances[c] = np.maximum(
            shrink * class_var + (1.0 - shrink) * global_var,
            floor,
        )

    predicted = np.argmax(ev.logits, axis=1)
    chosen_mean = means[predicted]
    chosen_var = variances[predicted]
    delta = eval_residual - chosen_mean

    quadratic = np.mean((delta * delta) / chosen_var, axis=1)
    log_volume = np.mean(np.log(chosen_var), axis=1)
    scores = quadratic + 0.05 * log_volume
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
