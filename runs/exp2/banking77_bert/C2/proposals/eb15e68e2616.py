import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS examples can be close to some class while still being atypical for the class selected by the classifier. Class-specific calibration of the predicted-class Mahalanobis distance accounts for unequal within-class spread."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)
    n_classes = train_dist.shape[1]

    labels = np.asarray(tr.labels, dtype=int)
    labels = np.clip(labels, 0, n_classes - 1)
    own_train_dist = train_dist[np.arange(labels.shape[0]), labels]

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    sums = np.bincount(labels, weights=own_train_dist, minlength=n_classes)
    means = sums / np.maximum(counts, 1.0)

    centered_sq = (own_train_dist - means[labels]) ** 2
    var_sums = np.bincount(labels, weights=centered_sq, minlength=n_classes)
    stds = np.sqrt(var_sums / np.maximum(counts - 1.0, 1.0))
    fallback = np.sqrt(np.mean(centered_sq) + 1e-8)
    stds = np.where(stds > 1e-6, stds, fallback)

    pred = np.argmax(ev.logits, axis=1)
    pred = np.clip(pred, 0, n_classes - 1)
    predicted_dist = eval_dist[np.arange(pred.shape[0]), pred]

    score_out = (predicted_dist - means[pred]) / (stds[pred] + 1e-8)
    return np.nan_to_num(score_out, nan=0.0, posinf=1e6, neginf=-1e6)
