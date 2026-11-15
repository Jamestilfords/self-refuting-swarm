FAMILY = "knn"
ARM = "ce"
RATIONALE = "Moderately large-k cosine neighbor distance in encoder space measures whether a query is supported by a stable local in-scope neighborhood rather than by one accidental close example. Pooled representations retain semantic information that can expose OOS queries despite confident classifier logits."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0005"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    return ctx.h.knn(tr.pooled, ev.pooled, k=40, l2=True)
