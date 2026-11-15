import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope representations should have a clearly preferred class prototype, whereas OOS items often lie comparably far from multiple class means. A small relative gap between the nearest and second-nearest class Mahalanobis distances is therefore more OOS-like."
SOURCE = "https://www.emergentmind.com/papers/2101.03778"
PARENTS = []
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.pooled)
    nearest_two = np.partition(distances, 1, axis=1)[:, :2]
    d1 = nearest_two[:, 0]
    d2 = nearest_two[:, 1]
    relative_gap = (d2 - d1) / (d2 + d1 + 1e-8)
    scores = -relative_gap
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
