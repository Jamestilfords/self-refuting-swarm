import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Near-neighbor support should agree across the classifier bottleneck and the normalized encoder representation for in-scope queries. A query that is locally isolated in both views receives a high two-view neighborhood-isolation score, while the geometric combination avoids reducing to either single-space kNN ranking."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    k_bottleneck = min(31, len(tr.feats))
    k_encoder = min(7, len(tr.pooled))

    d_bottleneck = ctx.h.knn(tr.feats, ev.feats, k=k_bottleneck, l2=False)
    d_encoder = ctx.h.knn(
        ctx.h.l2(tr.pooled),
        ctx.h.l2(ev.pooled),
        k=k_encoder,
        l2=False,
    )

    d_bottleneck = np.maximum(np.asarray(d_bottleneck, dtype=float), 0.0)
    d_encoder = np.maximum(np.asarray(d_encoder, dtype=float), 0.0)

    score_out = np.sqrt(
        np.log1p(d_bottleneck) * np.log1p(20.0 * d_encoder)
    )
    return np.nan_to_num(score_out, nan=0.0, posinf=1e6, neginf=0.0)
