import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "An in-scope query should have local support in both the bottleneck representation used by the classifier "
    "and the broader encoder representation. Multiplying train-calibrated neighbor-distance tail probabilities "
    "flags examples unsupported in either complementary feature space."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0055", "m_0056", "m_0057"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = 16

    d_feat_train = ctx.h.knn(tr.feats, tr.feats, k=k + 1, l2=True)
    d_feat_eval = ctx.h.knn(tr.feats, ev.feats, k=k, l2=True)

    d_pool_train = ctx.h.knn(tr.pooled, tr.pooled, k=k + 1, l2=True)
    d_pool_eval = ctx.h.knn(tr.pooled, ev.pooled, k=k, l2=True)

    n_feat = ctx.h.normalizer("ecdf", d_feat_train)
    n_pool = ctx.h.normalizer("ecdf", d_pool_train)

    p_feat = np.clip(n_feat(d_feat_eval), 0.0, 1.0)
    p_pool = np.clip(n_pool(d_pool_eval), 0.0, 1.0)

    return -np.log1p(-np.clip(p_feat * p_pool, 0.0, 1.0 - 1e-12))
