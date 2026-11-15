FAMILY = "fusion"
ARM = "edl"
RATIONALE = "ER-1 grid: min-class Mahalanobis on encoder output (PCA 128), coupled to the logits by qsigmoid x mul, EDL arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.pooled, ev.pooled
    g = h.gda(x_tr, tr.labels, pca=128, whiten=True)
    d_ev = g.min_mahalanobis_sq(x_ev)
    norm = h.normalizer("qsigmoid", -g.min_mahalanobis_sq(x_tr))
    p_hat = norm(-d_ev)
    return h.couple_mul(ev.logits, p_hat)
