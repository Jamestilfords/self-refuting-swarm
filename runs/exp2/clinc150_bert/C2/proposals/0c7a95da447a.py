import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query assigned confidently to a class can still be anomalous if it lies far from that "
    "predicted class's training distribution. Class-specific standardization accounts for "
    "different within-class spread while retaining predicted-label consistency."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.feats)
    eval_dist = g.class_mahalanobis_sq(ev.feats)

    labels = np.asarray(tr.labels, dtype=int)
    n_classes = train_dist.shape[1]
    own_train = train_dist[np.arange(labels.shape[0]), labels]

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    sums = np.bincount(labels, weights=own_train, minlength=n_classes)
    means = sums / np.maximum(counts, 1.0)

    centered_sq = (own_train - means[labels]) ** 2
    var_sums = np.bincount(labels, weights=centered_sq, minlength=n_classes)
    stds = np.sqrt(var_sums / np.maximum(counts - 1.0, 1.0))
    fallback = np.sqrt(np.mean(centered_sq) + 1e-8)
    stds = np.where(stds > 1e-6, stds, fallback)
    stds = np.maximum(stds, 1e-6)

    predicted = np.argmax(ev.logits, axis=1).astype(int)
    predicted = np.clip(predicted, 0, n_classes - 1)
    own_eval = eval_dist[np.arange(predicted.shape[0]), predicted]

    scores = (own_eval - means[predicted]) / stds[predicted]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
