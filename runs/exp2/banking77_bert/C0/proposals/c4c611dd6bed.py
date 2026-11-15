import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Class-mean directions represent the encoder subspace used to discriminate known intents. This scores how surprising the remaining, class-conditional nuisance component is after removing those discriminative directions, so OOS inputs with unfamiliar residual structure receive larger scores."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    classes, inverse = np.unique(labels, return_inverse=True)
    n_classes = classes.size
    dim = x_train.shape[1]

    counts = np.bincount(inverse, minlength=n_classes).astype(float)
    means = np.zeros((n_classes, dim), dtype=float)
    np.add.at(means, inverse, x_train)
    means /= np.maximum(counts[:, None], 1.0)

    global_mean = np.average(means, axis=0, weights=counts)
    between = (means - global_mean) * np.sqrt(counts[:, None] / np.maximum(counts.sum(), 1.0))

    _, singular_values, vt = np.linalg.svd(between, full_matrices=False)
    if singular_values.size:
        threshold = max(float(singular_values[0]) * 1e-7, 1e-12)
        rank = int(np.sum(singular_values > threshold))
        keep = min(max(rank, 1), 48)
        basis = vt[:keep].T
    else:
        basis = np.zeros((dim, 0), dtype=float)

    def residual(x):
        centered = x - global_mean
        if basis.shape[1]:
            return centered - (centered @ basis) @ basis.T
        return centered

    r_train = residual(x_train)
    r_eval = residual(x_eval)

    residual_means = np.zeros((n_classes, dim), dtype=float)
    np.add.at(residual_means, inverse, r_train)
    residual_means /= np.maximum(counts[:, None], 1.0)

    centered_train = r_train - residual_means[inverse]
    pooled_var = np.sum(centered_train * centered_train, axis=0)
    pooled_var /= max(float(x_train.shape[0] - n_classes), 1.0)
    pooled_var = np.maximum(pooled_var, 1e-8)

    class_var = np.zeros((n_classes, dim), dtype=float)
    np.add.at(class_var, inverse, centered_train * centered_train)
    class_var /= np.maximum(counts[:, None], 1.0)

    shrink = counts[:, None] / (counts[:, None] + 20.0)
    class_var = shrink * class_var + (1.0 - shrink) * pooled_var
    class_var = np.maximum(class_var, 1e-8)

    k_logits = tr.logits.shape[1]
    mean_by_logit = np.tile(np.zeros((1, dim), dtype=float), (k_logits, 1))
    var_by_logit = np.tile(pooled_var[None, :], (k_logits, 1))

    valid = (classes >= 0) & (classes < k_logits)
    mean_by_logit[classes[valid]] = residual_means[valid]
    var_by_logit[classes[valid]] = class_var[valid]

    predicted = np.argmax(np.asarray(ev.logits), axis=1)
    z = r_eval - mean_by_logit[predicted]
    surprise = np.mean((z * z) / var_by_logit[predicted], axis=1)

    return np.log1p(np.maximum(surprise, 0.0))
