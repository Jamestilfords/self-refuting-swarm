FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions capture subtle deviations that are suppressed by dominant semantic variation. OOS queries should have larger residual distance in these trailing covariance directions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    return partial.bottom(ev.pooled)
