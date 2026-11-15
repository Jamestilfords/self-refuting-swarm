import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope queries should align strongly with at least one class prototype in normalized bottleneck space. Low maximum cosine similarity indicates weak membership in every known intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels)
    classes, inverse = np.unique(labels, return_inverse=True)
    prototypes = np.zeros((classes.size, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, x_train)
    counts = np.bincount(inverse, minlength=classes.size).astype(float)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)
    max_cosine = np.max(x_eval @ prototypes.T, axis=1)
    return 1.0 - max_cosine
