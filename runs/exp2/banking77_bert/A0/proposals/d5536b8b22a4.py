FAMILY = "knn"
ARM = "ce"
RATIONALE = "The distance to a moderately distant neighbor in raw bottleneck space estimates local in-scope support while being less sensitive than the single nearest example to duplicates or accidental close matches. OOS queries should lie in lower-density regions and therefore have larger k-neighbor distances."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0047"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    return ctx.h.knn(tr.feats, ev.feats, k=32, l2=False)
