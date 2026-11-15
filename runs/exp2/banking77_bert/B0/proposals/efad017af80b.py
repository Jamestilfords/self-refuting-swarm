import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS queries often have weak class-conditional geometric preference: their predicted-class distance is "
    "similar to the runner-up class distance, or the classifier prediction disagrees with feature geometry. "
    "The predicted-to-runner-up Mahalanobis ratio captures this local intent-assignment instability."
)
SOURCE = "https://ar5iv.labs.arxiv.org/html/2203.07798"
PARENTS = []
MEMORY_USED = ["m_0047", "m_0060", "m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = np.maximum(g.class_mahalanobis_sq(ev.pooled), 0.0)

    predicted = np.argmax(ev.logits, axis=1)
    predicted = np.clip(predicted, 0, distances.shape[1] - 1)
    predicted_distance = distances[np.arange(distances.shape[0]), predicted]

    nearest_two = np.partition(distances, 1, axis=1)[:, :2]
    runner_up_distance = nearest_two[:, 1]

    scale = np.median(np.maximum(g.min_mahalanobis_sq(tr.pooled), 0.0))
    eps = max(float(scale) * 1e-8, 1e-12)
    scores = predicted_distance / (runner_up_distance + eps)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
