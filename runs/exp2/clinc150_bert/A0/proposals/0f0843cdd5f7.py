import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope representations should usually have a clearly preferred class prototype, "
    "whereas OOS queries often lie between several known intent regions and have a small nearest-versus-second-nearest distance gap."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0075"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    d = g.class_mahalanobis_sq(ev.pooled)
    nearest_two = np.partition(d, 1, axis=1)[:, :2]
    d1 = nearest_two[:, 0]
    d2 = nearest_two[:, 1]

    # Small relative prototype separation indicates ambiguous geometric support.
    separation = (d2 - d1) / (d2 + 1e-12)
    return 1.0 - separation
