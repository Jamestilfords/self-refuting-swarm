FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ER-1 grid: relative Mahalanobis on bottleneck features (PCA 128), coupled to the logits by qsigmoid x mul, CE arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.feats, ev.feats
    g = h.gda(x_tr, tr.labels, pca=128, whiten=True)
    d_ev = g.relative_mahalanobis_sq(x_ev)
    norm = h.normalizer("qsigmoid", -g.relative_mahalanobis_sq(x_tr))
    p_hat = norm(-d_ev)
    return h.couple_mul(ev.logits, p_hat)
