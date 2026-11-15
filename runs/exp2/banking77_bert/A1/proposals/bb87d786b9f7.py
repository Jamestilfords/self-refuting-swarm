import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Agreement between normalized encoder-space and bottleneck-space neighborhood distances reduces sensitivity to a single representation while detecting queries unsupported by local in-scope examples. Using a moderately large neighborhood emphasizes intent-region support rather than exact duplicate matching."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d_pooled = ctx.h.knn(tr.pooled, ev.pooled, k=35, l2=True)
    d_feats = ctx.h.knn(tr.feats, ev.feats, k=35, l2=True)
    return 0.5 * (np.log1p(np.maximum(d_pooled, 0.0)) +
                  np.log1p(np.maximum(d_feats, 0.0)))
