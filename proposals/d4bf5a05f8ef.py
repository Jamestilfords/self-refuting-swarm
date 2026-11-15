FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS representations can deviate strongly along low-variance within-class directions that are tightly constrained for in-scope intents. A trailing-eigenspace partial Mahalanobis score emphasizes these unsupported deviations."
SOURCE = "https://cdn.amazon.science/64/48/4d1e599c4472859e27a9f7647786/few-shot-out-of-domain-intent-detection-with-covariance-corrected-mahalanobis-distance.pdf"
PARENTS = []
MEMORY_USED = ["m_0002"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    return partial.bottom(ev.pooled)
