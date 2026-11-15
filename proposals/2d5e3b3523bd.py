FAMILY = "fusion"
ARM = "edl"
RATIONALE = "ER-1 grid: GDA log-density on encoder output (no PCA), coupled to the logits by ecdf x add_log, lambda 5, EDL arm."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.pooled, ev.pooled
    g = h.gda(x_tr, tr.labels, pca=None, whiten=True)
    d_ev = g.neg_log_density(x_ev)
    norm = h.normalizer("ecdf", -g.neg_log_density(x_tr))
    p_hat = norm(-d_ev)
    return h.couple_add_log(ev.logits, p_hat, 5.0)
