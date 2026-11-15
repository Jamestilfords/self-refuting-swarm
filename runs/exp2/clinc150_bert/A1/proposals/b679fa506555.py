import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration shape has characteristic entropy and dominance for each known intent. "
    "A class-conditioned distance in this shape space flags unusually uniform or unusually concentrated evidence allocations without being a monotone function of vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=np.float64)
        alpha = np.clip(alpha, 1e-10, 1e12)
        q = alpha / np.sum(alpha, axis=1, keepdims=True)
        entropy = -np.sum(q * np.log(q), axis=1) / np.log(q.shape[1])
        dominance = np.max(q, axis=1)
        return np.column_stack((entropy, dominance))

    train_shape = shape_features(tr.logits)
    eval_shape = shape_features(ev.logits)
    labels = np.asarray(tr.labels, dtype=np.int64)
    pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = np.mean(train_shape, axis=0)
    centered = train_shape - global_mean
    global_var = np.mean(centered * centered, axis=0) + 1e-6

    means = np.empty((n_classes, 2), dtype=np.float64)
    variances = np.empty((n_classes, 2), dtype=np.float64)

    for c in range(n_classes):
        members = train_shape[labels == c]
        if len(members) >= 2:
            means[c] = np.mean(members, axis=0)
            variances[c] = np.mean((members - means[c]) ** 2, axis=0)
        elif len(members) == 1:
            means[c] = members[0]
            variances[c] = global_var
        else:
            means[c] = global_mean
            variances[c] = global_var

    variances = np.maximum(variances, 0.05 * global_var)
    delta = eval_shape - means[pred]
    scores = np.sum((delta * delta) / variances[pred], axis=1)
    return np.asarray(scores, dtype=np.float64)
