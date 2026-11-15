import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope queries should align closely with at least one class prototype in normalized bottleneck feature space. OOS queries are expected to have lower maximum cosine similarity to all learned class directions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels
    classes, inverse = np.unique(labels, return_inverse=True)
    sums = np.zeros((classes.shape[0], x_train.shape[1]), dtype=float)
    np.add.at(sums, inverse, x_train)
    prototypes = ctx.h.l2(sums)
    max_cosine = np.max(x_eval @ prototypes.T, axis=1)
    return 1.0 - max_cosine
