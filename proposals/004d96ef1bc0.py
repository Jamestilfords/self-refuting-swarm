FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ER-1 grid: relative Mahalanobis on encoder output (PCA 128), coupled to the logits by ecdf x add_log, lambda 80, CE arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.pooled, ev.pooled
    g = h.gda(x_tr, tr.labels, pca=128, whiten=True)
    d_ev = g.relative_mahalanobis_sq(x_ev)
    norm = h.normalizer("ecdf", -g.relative_mahalanobis_sq(x_tr))
    p_hat = norm(-d_ev)
    return h.couple_add_log(ev.logits, p_hat, 80.0)
