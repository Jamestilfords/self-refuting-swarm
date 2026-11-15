import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope representations should usually have a clearly preferred Gaussian intent, while OOS inputs may be similarly distant from multiple intent prototypes. The negative gap between the nearest and second-nearest class Mahalanobis distances scores this prototype ambiguity, with a small distance term breaking ties for globally remote points."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.pooled)
    nearest_two = np.partition(distances, 1, axis=1)[:, :2]
    d_min = np.min(nearest_two, axis=1)
    d_second = np.max(nearest_two, axis=1)
    scores = -(d_second - d_min) + 0.01 * d_min
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
