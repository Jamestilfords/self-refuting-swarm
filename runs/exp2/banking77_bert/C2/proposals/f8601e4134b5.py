import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries are often similarly incompatible with several class regions, producing a small relative gap between their nearest and second-nearest class Mahalanobis distances. In-scope queries should more often have one distinctly preferred class region."
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
    return np.nan_to_num(-relative_gap, nan=0.0, posinf=0.0, neginf=0.0)
