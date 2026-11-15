import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "L2-normalized class prototypes capture angular intent structure independent of feature magnitude. OOS queries tend to have lower maximum cosine similarity to every train-derived intent prototype."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels
    classes = np.unique(labels)

    prototypes = np.empty((classes.size, x_train.shape[1]), dtype=float)
    for i, c in enumerate(classes):
        prototype = x_train[labels == c].mean(axis=0, keepdims=True)
        prototypes[i] = ctx.h.l2(prototype)[0]

    max_cosine = (x_eval @ prototypes.T).max(axis=1)
    return 1.0 - max_cosine
