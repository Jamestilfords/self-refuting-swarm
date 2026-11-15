FAMILY = "knn"
ARM = "ce"
RATIONALE = "OOS queries should be distant from training examples across both bottleneck and encoder representations. Averaging near- and broader-neighborhood radii reduces sensitivity to a single anomalously close neighbor while retaining local support information."
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0087", "m_0088"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k_small = min(8, len(tr.labels))
    k_large = min(32, len(tr.labels))

    pooled_small = ctx.h.knn(tr.pooled, ev.pooled, k=k_small, l2=True)
    pooled_large = ctx.h.knn(tr.pooled, ev.pooled, k=k_large, l2=True)
    feat_small = ctx.h.knn(tr.feats, ev.feats, k=k_small, l2=True)
    feat_large = ctx.h.knn(tr.feats, ev.feats, k=k_large, l2=True)

    return 0.25 * (
        pooled_small + pooled_large + feat_small + feat_large
    )
