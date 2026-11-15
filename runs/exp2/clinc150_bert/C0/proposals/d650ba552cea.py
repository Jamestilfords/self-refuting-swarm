import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries are often not clearly associated with a single class prototype, producing a small separation between their nearest and second-nearest class distances. The nearest-to-second-nearest distance ratio emphasizes this class-boundary ambiguity rather than overall density alone."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    train_two = np.partition(train_dist, 1, axis=1)[:, :2]
    train_scale = np.median(train_two[:, 1])
    eps = max(float(train_scale) * 1e-6, 1e-8)

    distances = g.class_mahalanobis_sq(ev.pooled)
    two_nearest = np.partition(distances, 1, axis=1)[:, :2]
    nearest = two_nearest[:, 0]
    second_nearest = two_nearest[:, 1]

    scores = nearest / (second_nearest + eps)
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
