import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Angular pooled representations from in-scope queries should have support at both very local and broader neighborhood scales. Averaging several L2-normalized kNN radii reduces sensitivity to an accidental close neighbor while retaining global semantic-manifold novelty."
SOURCE = "https://github.com/deeplearning-wisc/knn-ood/blob/master/run_imagenet.py"
PARENTS = []
MEMORY_USED = ["m_0056", "m_0057", "m_0123"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d4 = ctx.h.knn(tr.pooled, ev.pooled, k=4, l2=True)
    d16 = ctx.h.knn(tr.pooled, ev.pooled, k=16, l2=True)
    d64 = ctx.h.knn(tr.pooled, ev.pooled, k=64, l2=True)
    return (np.asarray(d4, dtype=float) +
            np.asarray(d16, dtype=float) +
            np.asarray(d64, dtype=float)) / 3.0
