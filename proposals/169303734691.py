FAMILY = "density"
ARM = "edl"
RATIONALE = "ER-1 grid: min-class Mahalanobis on encoder output (no PCA), EDL arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.pooled, ev.pooled
    g = h.gda(x_tr, tr.labels, pca=None, whiten=True)
    d_ev = g.min_mahalanobis_sq(x_ev)
    return d_ev
