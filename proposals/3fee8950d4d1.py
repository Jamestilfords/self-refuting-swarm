FAMILY = "density"
ARM = "ce"
RATIONALE = "ER-1 grid: GDA log-density on bottleneck features (PCA 128), CE arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.feats, ev.feats
    g = h.gda(x_tr, tr.labels, pca=128, whiten=True)
    d_ev = g.neg_log_density(x_ev)
    return d_ev
