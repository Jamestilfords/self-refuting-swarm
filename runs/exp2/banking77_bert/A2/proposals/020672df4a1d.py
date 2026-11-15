FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions encode constraints that in-scope representations satisfy tightly. OOS inputs should produce larger residuals in these trailing covariance directions even when their high-variance coordinates appear plausible."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    projector = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    return projector.bottom(ev.pooled)
