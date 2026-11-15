import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope representations should be substantially closer to one class Gaussian than to "
    "the runner-up class, whereas OOS queries are more likely to occupy geometrically ambiguous regions."
)
SOURCE = "https://arxiv.org/html/2510.15202v2"
PARENTS = []
MEMORY_USED = ["m_0086"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    distances = g.class_mahalanobis_sq(ev.pooled)
    nearest_two = np.partition(distances, 1, axis=1)[:, :2]
    d1 = nearest_two[:, 0]
    d2 = nearest_two[:, 1]

    relative_gap = (d2 - d1) / (d2 + d1 + 1e-12)
    return -np.nan_to_num(relative_gap, nan=0.0, posinf=0.0, neginf=0.0)
