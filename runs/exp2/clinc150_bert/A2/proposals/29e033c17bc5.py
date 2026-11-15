import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope embeddings usually have a clearly preferred Gaussian intent region, whereas OOS "
    "queries can be geometrically ambiguous between multiple class regions. A small gap between "
    "the nearest and second-nearest class Mahalanobis distances therefore indicates OOS risk."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    d = g.class_mahalanobis_sq(ev.pooled)
    two_smallest = np.partition(d, 1, axis=1)[:, :2]
    nearest = two_smallest[:, 0]
    second = two_smallest[:, 1]

    gap = second - nearest
    scale = np.median(
        np.partition(g.class_mahalanobis_sq(tr.pooled), 1, axis=1)[:, 1]
        - np.partition(g.class_mahalanobis_sq(tr.pooled), 1, axis=1)[:, 0]
    )
    scale = max(float(scale), 1e-6)

    return np.nan_to_num(-gap / scale, nan=0.0, posinf=1e6, neginf=-1e6)
