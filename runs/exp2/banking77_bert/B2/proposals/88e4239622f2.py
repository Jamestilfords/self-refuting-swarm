FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class directions constrain in-scope representations tightly, so OOS inputs should have larger residual deviation in the trailing covariance subspace. This targets information that can be obscured by dominant semantic variation."
SOURCE = "https://www.gatsby.ucl.ac.uk/~balaji/udl2021/accepted-papers/UDL2021-paper-007.pdf"
PARENTS = []
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    detector = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    return detector.bottom(ev.pooled)
