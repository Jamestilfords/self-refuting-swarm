import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A multi-scale cosine kNN distance in bottleneck space captures both local membership near training paraphrases and broader support around each known intent. Averaging short- and longer-range neighbor distances reduces sensitivity to isolated training examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    near = ctx.h.knn(tr.feats, ev.feats, k=8, l2=True)
    broad = ctx.h.knn(tr.feats, ev.feats, k=32, l2=True)
    return 0.5 * np.asarray(near) + 0.5 * np.asarray(broad)
