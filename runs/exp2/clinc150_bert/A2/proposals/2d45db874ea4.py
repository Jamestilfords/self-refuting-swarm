import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope representations should usually have a clearly preferred class centroid, whereas OOS queries can be similarly distant from multiple known intents. A small nearest-versus-second-nearest Mahalanobis gap therefore indicates geometric class ambiguity rather than merely global distance."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0033"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    def ambiguity(x):
        distances = g.class_mahalanobis_sq(x)
        two = np.partition(distances, 1, axis=1)[:, :2]
        nearest = np.min(two, axis=1)
        second = np.max(two, axis=1)
        return -(second - nearest) / (second + nearest + 1e-8)

    train_score = ambiguity(tr.pooled)
    eval_score = ambiguity(ev.pooled)

    ordered = np.sort(train_score)
    tail = np.searchsorted(ordered, eval_score, side="right") / float(len(ordered))

    center = np.median(train_score)
    scale = np.median(np.abs(train_score - center)) + 1e-8
    return tail + 1e-6 * (eval_score - center) / scale
