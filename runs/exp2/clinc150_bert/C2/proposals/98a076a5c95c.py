import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "OOS queries can have unusually large representation residuals after removing dominant training variation, especially relative to the centroid of their predicted intent. Class-conditional robust calibration prevents naturally broad in-scope intents from dominating the residual score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    n_train, dim = x_train.shape
    n_classes = tr.logits.shape[1]
    n_comp = max(1, min(96, dim - 1, n_train - 1))

    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=314159)
    pca.fit(x_train)
    basis = pca.components_

    global_center = np.mean(x_train, axis=0)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    sums = np.zeros((n_classes, dim), dtype=np.float64)
    np.add.at(sums, labels, x_train)
    centers = np.divide(
        sums,
        np.maximum(counts[:, None], 1.0),
        out=np.tile(global_center, (n_classes, 1)),
        where=counts[:, None] > 0,
    )

    train_delta = x_train - centers[labels]
    train_proj = train_delta @ basis.T
    train_residual = np.sqrt(np.maximum(
        np.sum(train_delta * train_delta, axis=1) - np.sum(train_proj * train_proj, axis=1),
        0.0,
    ))

    medians = np.zeros(n_classes, dtype=np.float64)
    scales = np.ones(n_classes, dtype=np.float64)
    global_median = float(np.median(train_residual))
    global_mad = float(np.median(np.abs(train_residual - global_median))) + 1e-8

    for c in range(n_classes):
        values = train_residual[labels == c]
        if values.size >= 5:
            medians[c] = np.median(values)
            scales[c] = np.median(np.abs(values - medians[c])) + 1e-8
        else:
            medians[c] = global_median
            scales[c] = global_mad

    predicted = np.argmax(np.asarray(ev.logits), axis=1)
    eval_delta = x_eval - centers[predicted]
    eval_proj = eval_delta @ basis.T
    eval_residual = np.sqrt(np.maximum(
        np.sum(eval_delta * eval_delta, axis=1) - np.sum(eval_proj * eval_proj, axis=1),
        0.0,
    ))

    scores = (eval_residual - medians[predicted]) / scales[predicted]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
