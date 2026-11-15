import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "This score measures disagreement between the classifier's predicted class and the representation-space class geometry, while also flagging samples with a small nearest-versus-second-nearest Gaussian distance margin. OOS inputs are often ambiguous in class-conditional feature space or assigned to a logit class that is not their nearest feature distribution."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_sorted = np.partition(train_d, 1, axis=1)
    train_gap = train_sorted[:, 1] - train_sorted[:, 0]
    gap_scale = np.median(train_gap[np.isfinite(train_gap)])
    if not np.isfinite(gap_scale) or gap_scale <= 1e-12:
        gap_scale = 1.0

    order = np.partition(eval_d, 1, axis=1)
    nearest = order[:, 0]
    second = order[:, 1]
    margin_ambiguity = np.exp(-(second - nearest) / gap_scale)

    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.clip(predicted, 0, eval_d.shape[1] - 1)
    predicted_distance = eval_d[np.arange(eval_d.shape[0]), predicted]

    disagreement = np.maximum(predicted_distance - nearest, 0.0)
    train_nearest = train_sorted[:, 0]
    distance_scale = np.median(train_nearest[np.isfinite(train_nearest)])
    if not np.isfinite(distance_scale) or distance_scale <= 1e-12:
        distance_scale = 1.0

    disagreement = 1.0 - np.exp(-disagreement / distance_scale)
    scores = 0.65 * margin_ambiguity + 0.35 * disagreement

    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
