import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A global bottleneck principal subspace removes common in-scope variation, while "
    "predicted-intent-specific calibration identifies unusually large unsupported residual components."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    dim = min(128, x_train.shape[1], max(1, x_train.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="full")
    pca.fit(x_train)

    def residual_norm(x):
        projected = pca.inverse_transform(pca.transform(x))
        return np.sqrt(np.sum((x - projected) ** 2, axis=1))

    train_residual = np.log1p(residual_norm(x_train))
    eval_residual = np.log1p(residual_norm(x_eval))

    global_center = np.median(train_residual)
    global_scale = np.median(np.abs(train_residual - global_center))
    global_scale = max(1.4826 * global_scale, 1e-6)

    n_classes = int(max(np.max(labels) + 1, tr.logits.shape[1]))
    centers = np.full(n_classes, global_center, dtype=np.float64)
    scales = np.full(n_classes, global_scale, dtype=np.float64)

    for c in range(n_classes):
        values = train_residual[labels == c]
        if values.size >= 3:
            center = np.median(values)
            scale = 1.4826 * np.median(np.abs(values - center))
            centers[c] = center
            scales[c] = max(scale, 0.25 * global_scale, 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.clip(predicted, 0, n_classes - 1)
    return (eval_residual - centers[predicted]) / scales[predicted]
