FAMILY = "fusion"
ARM = "edl"
RATIONALE = "ER-1 grid: partial Mahalanobis on encoder output (no PCA), coupled to the logits by qsigmoid x mul, EDL arm."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev, h = ctx.train, ctx.eval, ctx.h
    x_tr, x_ev = tr.pooled, ev.pooled
    g = h.partial(x_tr, tr.labels, top_k=128)
    d_ev = g.bottom(x_ev)
    norm = h.normalizer("qsigmoid", -g.bottom(x_tr))
    p_hat = norm(-d_ev)
    return h.couple_mul(ev.logits, p_hat)
