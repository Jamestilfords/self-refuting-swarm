import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query whose classifier-routed intent is substantially farther in representation space "
    "than an alternative intent exhibits logit-feature disagreement that is uncommon for in-scope inputs."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0133", "m_0137"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    def routed_gap(x, logits):
        distances = np.asarray(g.class_mahalanobis_sq(x), dtype=float)
        routed = np.argmax(logits, axis=1)
        routed_distance = distances[np.arange(distances.shape[0]), routed]
        alternatives = distances.copy()
        alternatives[np.arange(alternatives.shape[0]), routed] = np.inf
        nearest_alternative = np.min(alternatives, axis=1)
        return routed_distance - nearest_alternative

    train_gap = routed_gap(tr.pooled, tr.logits)
    eval_gap = routed_gap(ev.pooled, ev.logits)

    center = np.median(train_gap)
    scale = np.median(np.abs(train_gap - center))
    scale = max(float(scale), 1e-6)

    scores = (eval_gap - center) / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
