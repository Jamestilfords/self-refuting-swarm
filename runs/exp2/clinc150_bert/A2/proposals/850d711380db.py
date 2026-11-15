FAMILY = "density"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions emphasize low-variance features that in-scope examples tightly constrain. OOS queries can have unusually large residual energy in these directions even when their dominant semantic components resemble an intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    p = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    return p.bottom(ev.pooled)
