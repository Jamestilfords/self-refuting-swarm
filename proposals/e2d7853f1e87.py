FAMILY = "density"
ARM = "edl"
RATIONALE = "ER-1 grid: partial Mahalanobis on bottleneck features (PCA 128), EDL arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.feats, ev.feats
    g = h.partial(x_tr, tr.labels, top_k=128)
    d_ev = g.top(x_ev)
    return d_ev
