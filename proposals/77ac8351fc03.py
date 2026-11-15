FAMILY = "fusion"
ARM = "edl"
RATIONALE = "ER-1 grid: relative Mahalanobis on encoder output (no PCA), coupled to the logits by minmax x mul, EDL arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.pooled, ev.pooled
    g = h.gda(x_tr, tr.labels, pca=None, whiten=True)
    d_ev = g.relative_mahalanobis_sq(x_ev)
    norm = h.normalizer("minmax", -g.relative_mahalanobis_sq(x_tr))
    p_hat = norm(-d_ev)
    return h.couple_mul(ev.logits, p_hat)
