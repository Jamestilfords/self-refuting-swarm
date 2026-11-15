FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ER-1 grid: min-class Mahalanobis on bottleneck features (no PCA), coupled to the logits by qsigmoid x add_log, lambda 1, CE arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.feats, ev.feats
    g = h.gda(x_tr, tr.labels, pca=None, whiten=True)
    d_ev = g.min_mahalanobis_sq(x_ev)
    norm = h.normalizer("qsigmoid", -g.min_mahalanobis_sq(x_tr))
    p_hat = norm(-d_ev)
    return h.couple_add_log(ev.logits, p_hat, 1.0)
