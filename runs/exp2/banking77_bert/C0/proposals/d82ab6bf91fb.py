FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs can deviate along low-variance within-class directions that are suppressed by standard leading-component representations. Trailing eigendirections emphasize these subtle class-conditional residual shifts."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    residual = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    scores = residual.bottom(ev.pooled)
    return scores
