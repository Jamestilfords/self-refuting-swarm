import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope representations should be substantially closer to one intent mean than to the "
    "next-best intent mean, whereas OOS queries often lie in ambiguous regions between intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.pooled)

    if distances.shape[1] < 2:
        scores = distances[:, 0]
    else:
        two_smallest = np.partition(distances, 1, axis=1)[:, :2]
        nearest = two_smallest[:, 0]
        second_nearest = two_smallest[:, 1]
        scores = nearest - second_nearest

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
