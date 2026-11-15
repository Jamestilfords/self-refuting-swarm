import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query whose classifier-predicted class is geometrically implausible is likely OOS, even when it lies near a different known class. The predicted-class Mahalanobis excess measures this classifier-feature disagreement."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    d = g.class_mahalanobis_sq(ev.pooled)
    pred = np.argmax(ev.logits, axis=1)
    predicted_distance = d[np.arange(d.shape[0]), pred]
    nearest_distance = np.min(d, axis=1)

    scores = predicted_distance - nearest_distance
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
