import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries often lack a clearly dominant class-conditional feature match, producing a small gap between their nearest and second-nearest Gaussian class distances. This boundary-ambiguity score differs from minimum-distance density by emphasizing competition among classes rather than absolute proximity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    d = g.class_mahalanobis_sq(ev.pooled)
    two = np.partition(d, 1, axis=1)[:, :2]
    nearest = two[:, 0]
    second = two[:, 1]

    gap = (second - nearest) / (np.abs(second) + 1.0)
    scores = -gap
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
