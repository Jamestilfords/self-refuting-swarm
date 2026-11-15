import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS queries can create disagreement between the classifier's predicted class and the closest "
    "class in feature space. The excess Mahalanobis distance of the predicted class over its best competitor measures this disagreement."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)
    n_classes = train_dist.shape[1]

    def disagreement(dist, logits):
        pred = np.argmax(logits, axis=1)
        pred = np.clip(pred, 0, n_classes - 1)
        rows = np.arange(dist.shape[0])
        predicted = dist[rows, pred]

        masked = dist.copy()
        masked[rows, pred] = np.inf
        competitor = np.min(masked, axis=1)
        return predicted - competitor, pred

    train_score, train_pred = disagreement(train_dist, tr.logits)
    eval_score, eval_pred = disagreement(eval_dist, ev.logits)

    centers = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)
    for c in range(n_classes):
        values = train_score[train_pred == c]
        if values.size:
            center = np.median(values)
            mad = np.median(np.abs(values - center))
            centers[c] = center
            scales[c] = max(1.4826 * mad, 1e-6)

    scores = (eval_score - centers[eval_pred]) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
