import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A prediction is unreliable when its bottleneck representation lies beyond the "
    "class-specific support radius of the intent selected by the classifier. "
    "Class-adaptive centroids and radii account for intents with naturally different spread."
)
SOURCE = "https://hanleizhang.github.io/files/TASLP23-DA-ADB/DA-ADB.pdf"
PARENTS = []
MEMORY_USED = ["m_0133"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    centers = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    radii = np.ones(n_classes, dtype=float)

    for c in range(n_classes):
        xc = x_train[labels == c]
        if len(xc) == 0:
            centers[c] = x_train.mean(axis=0)
            continue
        center = xc.mean(axis=0)
        centers[c] = center
        distances = np.sqrt(np.sum((xc - center) ** 2, axis=1))
        radius = np.quantile(distances, 0.90)
        radii[c] = max(float(radius), 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    residual = x_eval - centers[predicted]
    distances = np.sqrt(np.sum(residual * residual, axis=1))
    return distances / radii[predicted] - 1.0
