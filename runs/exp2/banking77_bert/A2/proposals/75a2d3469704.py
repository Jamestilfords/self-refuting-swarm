import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS queries can be both far from all known intents and geometrically ambiguous between "
    "multiple class means. A train-calibrated ratio of nearest-class distance to the nearest-vs-runner-up gap emphasizes this joint failure mode."
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

    train_gap = np.maximum(train_two[:, 1] - train_two[:, 0], 0.0)
    gap_floor = float(np.quantile(train_gap, 0.25))
    scale = float(np.median(train_two[:, 0]))
    eps = max(1e-8, 1e-6 * max(scale, 1.0))

    nearest = np.maximum(eval_two[:, 0], 0.0)
    gap = np.maximum(eval_two[:, 1] - eval_two[:, 0], 0.0)
    return nearest / (gap + gap_floor + eps)
