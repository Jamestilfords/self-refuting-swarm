import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multiscale normalized kNN distance estimates both very local and broader in-scope support, reducing sensitivity to an accidental single close training neighbor while flagging queries that remain isolated across neighborhood sizes."
SOURCE = "https://icml.cc/media/icml-2022/Slides/16494.pdf"
PARENTS = []
MEMORY_USED = ["m_0088"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    d4 = ctx.h.knn(tr.pooled, ev.pooled, k=4, l2=True)
    d16 = ctx.h.knn(tr.pooled, ev.pooled, k=16, l2=True)
    d64 = ctx.h.knn(tr.pooled, ev.pooled, k=64, l2=True)
    out = 0.20 * d4 + 0.35 * d16 + 0.45 * d64
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
