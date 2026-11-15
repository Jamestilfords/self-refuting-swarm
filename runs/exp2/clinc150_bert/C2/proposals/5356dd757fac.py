FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs can deviate along low-variance within-class directions that are suppressed by full-space distance measures. Trailing-eigenspace residuals emphasize these locally constrained directions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    trailing = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    return trailing.bottom(ev.pooled)
