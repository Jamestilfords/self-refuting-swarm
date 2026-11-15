import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope embeddings should be decisively closer to one intent prototype than to competing intent prototypes. OOS inputs are often forced between or beyond intent regions, yielding a small relative gap between their nearest and runner-up class distances."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0041"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.pooled)

    nearest_two = np.partition(distances, kth=1, axis=1)[:, :2]
    d1 = np.min(nearest_two, axis=1)
    d2 = np.max(nearest_two, axis=1)

    separation = (d2 - d1) / (d2 + d1 + 1e-12)
    scores = 1.0 - separation
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
