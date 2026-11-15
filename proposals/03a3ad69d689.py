import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query can be close to some intent cluster while being inconsistent with the class predicted by its logits. "
    "The predicted-class Mahalanobis distance flags these classifier-density disagreements as out-of-scope."
)
SOURCE = "https://cdn.amazon.science/64/48/4d1e599c4472859e27a9f7647786/few-shot-out-of-domain-intent-detection-with-covariance-corrected-mahalanobis-distance.pdf"
PARENTS = []
MEMORY_USED = ["m_0002"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.pooled)
    predicted = np.argmax(ev.logits, axis=1)
    return distances[np.arange(ev.logits.shape[0]), predicted]
