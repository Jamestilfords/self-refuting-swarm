FAMILY = "density"
ARM = "edl"
RATIONALE = "ER-1 grid: GDA log-density on bottleneck features (no PCA), EDL arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.feats, ev.feats
    g = h.gda(x_tr, tr.labels, pca=None, whiten=True)
    d_ev = g.neg_log_density(x_ev)
    return d_ev
