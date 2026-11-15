import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = "A class-conditional residual score measures how far an item lies outside the shared low-dimensional within-intent variation subspace after centering on its predicted intent. OOS queries can be close to the global feature manifold while still having unusually large residual variation relative to every intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]
    dim = min(96, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    sums = np.zeros((n_classes, tr.pooled.shape[1]), dtype=float)
    np.add.at(sums, labels, tr.pooled)
    global_mean = np.mean(tr.pooled, axis=0)

    means = np.tile(global_mean, (n_classes, 1))
    present = counts > 0
    means[present] = sums[present] / counts[present, None]

    train_centered = tr.pooled - means[labels]
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=9137)
    pca.fit(train_centered)
    components = pca.components_

    train_proj = train_centered @ components.T
    train_resid_sq = np.maximum(
        np.sum(train_centered * train_centered, axis=1)
        - np.sum(train_proj * train_proj, axis=1),
        0.0,
    )
    train_resid = np.sqrt(train_resid_sq)

    class_med = np.zeros(n_classes, dtype=float)
    class_scale = np.zeros(n_classes, dtype=float)
    global_med = float(np.median(train_resid))
    global_scale = float(np.median(np.abs(train_resid - global_med)) + 1e-6)

    for c in range(n_classes):
        values = train_resid[labels == c]
        if values.size >= 4:
            med = float(np.median(values))
            class_med[c] = med
            class_scale[c] = float(np.median(np.abs(values - med)) + 1e-6)
        else:
            class_med[c] = global_med
            class_scale[c] = global_scale

    predicted = np.argmax(ev.logits, axis=1)
    ev_centered = ev.pooled - means[predicted]
    ev_proj = ev_centered @ components.T
    ev_resid_sq = np.maximum(
        np.sum(ev_centered * ev_centered, axis=1)
        - np.sum(ev_proj * ev_proj, axis=1),
        0.0,
    )
    ev_resid = np.sqrt(ev_resid_sq)

    scores = (ev_resid - class_med[predicted]) / class_scale[predicted]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
