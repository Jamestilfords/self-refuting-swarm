import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "An in-scope item should be geometrically closest to the same class selected by its logits. "
    "The excess Mahalanobis distance of the predicted class over the nearest class detects disagreement between classifier intent evidence and class-conditional feature support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0036", "m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.pooled)

    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.minimum(predicted, distances.shape[1] - 1)
    own_distance = distances[np.arange(distances.shape[0]), predicted]
    nearest_distance = np.min(distances, axis=1)

    disagreement = own_distance - nearest_distance
    return np.nan_to_num(disagreement, nan=0.0, posinf=1e12, neginf=0.0)
