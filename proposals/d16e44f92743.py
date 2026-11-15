FAMILY = "fusion"
ARM = "ce"
RATIONALE = "ER-1 grid: partial Mahalanobis on encoder output (PCA 128), coupled to the logits by ecdf x add_log, lambda 160, CE arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.pooled, ev.pooled
    g = h.partial(x_tr, tr.labels, top_k=128)
    d_ev = g.top(x_ev)
    norm = h.normalizer("ecdf", -g.top(x_tr))
    p_hat = norm(-d_ev)
    return h.couple_add_log(ev.logits, p_hat, 160.0)
