import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope examples tend to be substantially closer to one class distribution than to competing class distributions. OOS examples often have a small nearest-versus-second-nearest class distance gap, indicating weak class-specific support."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.pooled)

    nearest_two = np.partition(distances, 1, axis=1)[:, :2]
    d1 = nearest_two[:, 0]
    d2 = nearest_two[:, 1]

    relative_gap = (d2 - d1) / (d1 + d2 + 1e-12)
    return -relative_gap
