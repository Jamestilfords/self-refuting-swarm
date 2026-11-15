FAMILY = "fusion"
ARM = "edl"
RATIONALE = "ER-1 grid: Mahalanobis++ on encoder output (PCA 128), coupled to the logits by ecdf x add_log, lambda 80, EDL arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = h.l2(tr.pooled), h.l2(ev.pooled)
    g = h.gda(x_tr, tr.labels, pca=128, whiten=True)
    d_ev = g.min_mahalanobis_sq(x_ev)
    norm = h.normalizer("ecdf", -g.min_mahalanobis_sq(x_tr))
    p_hat = norm(-d_ev)
    return h.couple_add_log(ev.logits, p_hat, 80.0)
