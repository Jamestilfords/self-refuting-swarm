import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope representations tend to have a clearly preferred class centroid, whereas OOS queries are often both far from known centroids and geometrically ambiguous between multiple intents. A nearest-versus-second-nearest Mahalanobis margin supplements absolute representation atypicality without relying on classifier logits."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    scale = np.median(train_two[:, 0])
    scale = max(float(scale), 1e-8)

    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]
    nearest = eval_two[:, 0]
    margin = eval_two[:, 1] - eval_two[:, 0]

    return (nearest - 0.5 * margin) / scale
