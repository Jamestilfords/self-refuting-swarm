import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs may be assigned a logit-predicted class that is not the closest class under the encoder feature distribution. The ratio between predicted-class distance and nearest-class distance measures this classifier-density disagreement rather than absolute density alone."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.pooled)
    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.clip(predicted, 0, distances.shape[1] - 1)

    d_pred = distances[np.arange(distances.shape[0]), predicted]
    d_near = np.min(distances, axis=1)
    disagreement = np.log1p(np.maximum(d_pred, 0.0)) - np.log1p(
        np.maximum(d_near, 0.0)
    )

    entropy = ctx.h.entropy(ev.logits)
    return disagreement + 0.05 * entropy
