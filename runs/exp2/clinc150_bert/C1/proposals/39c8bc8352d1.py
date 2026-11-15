import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS representations often lack a clearly dominant class-conditional Gaussian, producing a small margin between their nearest and second-nearest class distances. Normalizing this ambiguity by the nearest distance reduces sensitivity to global representation scale."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.pooled)
    two_nearest = np.partition(distances, 1, axis=1)[:, :2]
    nearest = two_nearest[:, 0]
    margin = np.maximum(two_nearest[:, 1] - nearest, 0.0)
    return nearest / (margin + 1e-6)
