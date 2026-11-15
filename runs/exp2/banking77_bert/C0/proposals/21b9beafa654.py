import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope representations should have a clearly preferred class centroid, while OOS points often lie in regions where the two closest class Gaussians are similarly plausible. The negative relative gap between the nearest and second-nearest class distances therefore assigns higher scores to ambiguous, boundary-like representations."
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
    return -(d2 - d1) / (d2 + d1 + 1e-12)
