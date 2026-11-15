import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope examples should align strongly with at least one class prototype in normalized bottleneck feature space, while OOS queries tend to have lower maximum cosine similarity."
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
        prototype = x_train[labels == c].mean(axis=0)
        norm = np.linalg.norm(prototype)
        prototypes[i] = prototype / max(norm, 1e-12)

    max_cosine = np.max(x_eval @ prototypes.T, axis=1)
    return 1.0 - max_cosine
