import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A moderately large k-nearest-neighbour radius in the 256-dimensional bottleneck feature space measures local in-scope support while being less sensitive to duplicate examples than very small k. OOS queries should typically require a larger neighbourhood to reach known-intent training examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0008"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    s = ctx.h.knn(tr.feats, ev.feats, k=75, l2=True)
    return np.nan_to_num(s, nan=0.0, posinf=1e12, neginf=-1e12)
