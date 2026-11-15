FAMILY = "density"
ARM = "ce"
RATIONALE = "ER-1 grid: Mahalanobis++ on bottleneck features (no PCA), CE arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = h.l2(tr.feats), h.l2(ev.feats)
    g = h.gda(x_tr, tr.labels, pca=None, whiten=True)
    d_ev = g.min_mahalanobis_sq(x_ev)
    return d_ev
