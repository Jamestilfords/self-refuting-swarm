import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS representations are often similarly far from several known intent centroids, producing a small "
    "nearest-versus-second-nearest Mahalanobis margin even when the classifier assigns a confident intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]

    train_margin = train_two[:, 1] - train_two[:, 0]
    eval_margin = eval_two[:, 1] - eval_two[:, 0]

    center = np.median(train_margin)
    scale = np.median(np.abs(train_margin - center)) + 1e-6

    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.minimum(predicted, eval_d.shape[1] - 1)
    pred_distance = eval_d[np.arange(eval_d.shape[0]), predicted]

    train_pred = np.argmax(tr.logits, axis=1)
    train_pred = np.minimum(train_pred, train_d.shape[1] - 1)
    train_pred_distance = train_d[np.arange(train_d.shape[0]), train_pred]
    distance_center = np.median(train_pred_distance)
    distance_scale = np.median(
        np.abs(train_pred_distance - distance_center)
    ) + 1e-6

    ambiguity = -(eval_margin - center) / scale
    pred_atypicality = (pred_distance - distance_center) / distance_scale
    scores = ambiguity + 0.25 * pred_atypicality

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
