import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope representations should have a clearly preferred class prototype, whereas OOS queries often lie in regions with similarly plausible nearby classes. The normalized gap between the nearest and second-nearest class Mahalanobis distances measures this prototype ambiguity rather than overall distance alone."
SOURCE = "https://github.com/kimjeyoung/Mahalanobis-BERT"
PARENTS = []
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)

    if distances.shape[1] < 2:
        return np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    nearest_two = np.partition(distances, 1, axis=1)[:, :2]
    d1 = nearest_two[:, 0]
    d2 = nearest_two[:, 1]
    margin = (d2 - d1) / (d2 + d1 + 1e-12)

    return np.nan_to_num(-margin, nan=0.0, posinf=0.0, neginf=0.0)
